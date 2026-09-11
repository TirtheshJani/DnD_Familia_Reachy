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
