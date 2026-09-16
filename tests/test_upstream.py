import json
import pytest
from familiar_brain import tools
from familiar_brain.tools import roll as roll_tool
from familiar_brain.upstream import Upstream, MAX_TOOL_ROUNDS
from tests.fakes.llm_server import FakeLLM

ROLL_ARGS = '{"expression":"1d20"}'


@pytest.fixture
def fake():
    f = FakeLLM(); yield f; f.close()


def _events(frames: list[str]) -> list[dict]:
    return [json.loads(f.split("data: ", 1)[1]) for f in frames if f.startswith("event:")]


def test_passthrough_text_stream(fake):
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names=set())
    frames = list(up.stream({"input": [{"role": "user", "content": "hi"}], "stream": True}))
    types = [e["type"] for e in _events(frames)]
    assert "response.output_text.delta" in types and types[-1] == "response.completed"
    assert fake.requests[-1]["body"]["id_slot"] == 0


def test_brain_tool_is_executed_and_round_tripped(fake):
    tools.register(roll_tool)
    call = {"id": "fc1", "type": "function_call", "call_id": "c1", "name": "roll",
            "arguments": ROLL_ARGS, "status": "completed"}
    fake.responses_events.append([
        {"type": "response.created", "response": {"id": "r1", "output": []}},
        {"type": "response.output_item.done", "output_index": 0, "item": call},
        {"type": "response.completed", "response": {"id": "r1", "output": [call]}},
    ])
    fake.responses_events.append([
        {"type": "response.output_text.delta", "delta": "You rolled well."},
        {"type": "response.completed", "response": {"id": "r2", "output": []}},
    ])
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names={"roll"})
    frames = list(up.stream({"input": [{"role": "user", "content": "roll a d20"}], "stream": True}))
    evs = _events(frames)
    assert not any(e.get("item", {}).get("name") == "roll" for e in evs), "brain tool call must not reach the client"
    assert any(e.get("delta") == "You rolled well." for e in evs)
    second = fake.requests[-1]["body"]["input"]
    assert second[-1]["type"] == "function_call_output" and second[-1]["call_id"] == "c1" and "Rolled 1d20" in second[-1]["output"]


# --- deviation tests, see docs/deviations.md ---

def test_brain_tool_argument_frames_do_not_leak(fake):
    """A real llama-server streams output_item.added, then function_call_arguments
    delta and done, then output_item.done. The middle two identify their item by
    item_id rather than by a nested item object, so a hide check that only inspects
    the item key lets them through and the robot sees argument frames for a tool it
    has no implementation for."""
    tools.register(roll_tool)
    call = {"id": "fc1", "type": "function_call", "call_id": "call_1", "name": "roll",
            "arguments": ROLL_ARGS, "status": "completed"}
    fake.responses_events.append([
        {"type": "response.created", "response": {"id": "r1", "output": []}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {**call, "status": "in_progress", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 0,
         "item_id": "fc1", "delta": ROLL_ARGS},
        {"type": "response.function_call_arguments.done", "output_index": 0,
         "item_id": "fc1", "arguments": ROLL_ARGS},
        {"type": "response.output_item.done", "output_index": 0, "item": call},
        {"type": "response.completed", "response": {"id": "r1", "output": [call]}},
    ])
    fake.responses_events.append([
        {"type": "response.output_text.delta", "delta": "Seventeen."},
        {"type": "response.completed", "response": {"id": "r2", "output": []}},
    ])
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names={"roll"})
    evs = _events(list(up.stream({"input": [{"role": "user", "content": "roll"}], "stream": True})))
    leaked = [e["type"] for e in evs if e.get("item_id") == "fc1"]
    assert leaked == [], f"brain tool argument frames leaked to the client: {leaked}"


def test_reasoning_effort_defaults_to_none_upstream(fake):
    """Gemma 4 is a reasoning model. Left to itself it spends the token budget
    thinking and then answers in prose instead of calling the tool, which both breaks
    dice rolling and blows the latency ceiling. Measured against the live model:
    reasoning effort none yields a bare function_call and no reasoning item."""
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names=set())
    list(up.stream({"input": [{"role": "user", "content": "hi"}], "stream": True}))
    assert fake.requests[-1]["body"]["reasoning"] == {"effort": "none"}


def test_explicit_reasoning_from_the_client_is_preserved(fake):
    """speech-to-speech exposes --responses_api_reasoning_effort, so a caller that
    sets it deliberately must win over the brain's default."""
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names=set())
    list(up.stream({"input": [{"role": "user", "content": "hi"}], "stream": True,
                    "reasoning": {"effort": "medium"}}))
    assert fake.requests[-1]["body"]["reasoning"] == {"effort": "medium"}


def test_stream_always_terminates_with_completed(fake):
    """If the model keeps calling a brain tool until the round budget is spent, the
    client must still receive a terminating response.completed rather than a
    truncated stream."""
    tools.register(roll_tool)
    call = {"id": "fc1", "type": "function_call", "call_id": "call_1", "name": "roll",
            "arguments": ROLL_ARGS, "status": "completed"}
    for _ in range(MAX_TOOL_ROUNDS + 1):
        fake.responses_events.append([
            {"type": "response.created", "response": {"id": "r1", "output": []}},
            {"type": "response.output_item.done", "output_index": 0, "item": call},
            {"type": "response.completed", "response": {"id": "r1", "output": [call]}},
        ])
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names={"roll"})
    evs = _events(list(up.stream({"input": [{"role": "user", "content": "roll"}], "stream": True})))
    assert evs, "stream produced no events at all"
    assert evs[-1]["type"] == "response.completed", f"stream ended on {evs[-1]['type']}"


# --- an addressed turn always gets words, see familiar_brain/app.py responses() ---

def _silent(name: str = "play_emotion", call_id: str = "call_g") -> list[dict]:
    call = {"id": f"fc_{call_id}", "type": "function_call", "call_id": call_id, "name": name,
            "arguments": "{}", "status": "completed"}
    return [{"type": "response.created", "response": {"id": "r1", "output": []}},
            {"type": "response.output_item.done", "output_index": 0, "item": call},
            {"type": "response.completed", "response": {"id": "r1", "output": [call]}}]


def _spoken(text: str = "Seventeen.") -> list[dict]:
    return [{"type": "response.output_text.delta", "delta": text},
            {"type": "response.completed", "response": {"id": "r2", "output": []}}]


def _responses_bodies(fake) -> list[dict]:
    return [r["body"] for r in fake.requests if r["path"].endswith("/responses")]


APP_TOOLS = [{"type": "function", "name": "camera"}, {"type": "function", "name": "roll"}]


def test_a_turn_that_must_speak_and_came_back_silent_is_asked_again_without_tools(fake):
    fake.responses_events.extend([_silent(), _spoken("Content enough.")])
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names=set())
    evs = _events(list(up.stream({"input": [{"role": "user", "content": "Rocky, how are you?"}],
                                  "tools": APP_TOOLS, "tool_choice": "auto", "stream": True},
                                 must_speak=True)))
    bodies = _responses_bodies(fake)
    assert len(bodies) == 2
    assert "tools" in bodies[0] and "tools" not in bodies[1] and "tool_choice" not in bodies[1]
    assert bodies[1]["input"] == bodies[0]["input"]
    assert any(e.get("delta") == "Content enough." for e in evs)
    # One completed, at the end: speech-to-speech ends the turn on the first it sees.
    assert [e["type"] for e in evs].count("response.completed") == 1 and evs[-1]["type"] == "response.completed"


def test_a_silent_turn_is_asked_again_only_once(fake):
    fake.responses_events.extend([_silent(), _silent(call_id="call_h")])
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names=set())
    evs = _events(list(up.stream({"input": [{"role": "user", "content": "Rocky?"}], "stream": True},
                                 must_speak=True)))
    assert len(_responses_bodies(fake)) == 2
    assert evs[-1]["type"] == "response.completed"


def test_a_camera_call_is_an_answer_and_needs_no_second_ask(fake):
    """The words come on the sight continuation, after the robot has taken the picture."""
    fake.responses_events.append(_silent("camera", "call_cam"))
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names=set())
    list(up.stream({"input": [{"role": "user", "content": "Rocky, what do you see?"}], "stream": True},
                   must_speak=True))
    assert len(_responses_bodies(fake)) == 1


def test_without_must_speak_a_silent_turn_goes_out_as_it_came(fake):
    fake.responses_events.append(_silent())
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names=set())
    evs = _events(list(up.stream({"input": [{"role": "user", "content": "hi"}], "stream": True})))
    assert len(_responses_bodies(fake)) == 1 and evs[-1]["type"] == "response.completed"


def test_the_second_ask_keeps_the_brain_tool_results(fake):
    """Asked again from the original request, the model would have lost the roll and be
    left to invent a number, which the persona forbids."""
    tools.register(roll_tool)
    call = {"id": "fc1", "type": "function_call", "call_id": "c1", "name": "roll",
            "arguments": ROLL_ARGS, "status": "completed"}
    fake.responses_events.extend([
        [{"type": "response.output_item.done", "output_index": 0, "item": call},
         {"type": "response.completed", "response": {"id": "r1", "output": [call]}}],
        [{"type": "response.completed", "response": {"id": "r2", "output": []}}],
        _spoken(),
    ])
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names={"roll"})
    evs = _events(list(up.stream({"input": [{"role": "user", "content": "Rocky, roll a d20"}],
                                  "tools": APP_TOOLS, "stream": True}, must_speak=True)))
    bodies = _responses_bodies(fake)
    assert len(bodies) == 3 and "tools" in bodies[1] and "tools" not in bodies[2]
    assert bodies[2]["input"][-1]["type"] == "function_call_output" and bodies[2]["input"][-1]["call_id"] == "c1"
    assert any(e.get("delta") == "Seventeen." for e in evs)
