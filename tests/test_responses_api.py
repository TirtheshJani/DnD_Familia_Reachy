import json
from familiar_brain.responses_api import (
    latest_user_text, function_call_outputs, sse, idle_response_events, with_instructions_and_tools,
    idle_response_json, wants_stream,
    call_item, item_events, tool_call_events, empty_response_events, idle_items,
    continuation_kind, GREETING_MARKER,
)

REQ = {
    "model": "familiar",
    "instructions": "old",
    "stream": True,
    "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "the door creaks"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "..."}]},
        {"role": "user", "content": "Understudy, roll a d20"},
    ],
    "tools": [{"type": "function", "name": "play_emotion", "parameters": {"type": "object", "properties": {}}}],
}


def test_latest_user_text_handles_string_and_parts():
    assert latest_user_text(REQ) == "Understudy, roll a d20"
    req2 = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]}
    assert latest_user_text(req2) == "hi"


def test_function_call_outputs_extracts_pending_results():
    req = {"input": [{"type": "function_call_output", "call_id": "c1", "output": "Rolled 1d20: 17"}]}
    assert function_call_outputs(req) == [("c1", "Rolled 1d20: 17")]


def test_sse_frames_event():
    frame = sse({"type": "response.completed", "response": {"id": "r"}})
    assert frame.startswith("event: response.completed\ndata: ") and frame.endswith("\n\n")
    assert json.loads(frame.split("data: ", 1)[1].strip())["type"] == "response.completed"


def test_idle_response_events_call_idle_do_nothing():
    evs = idle_response_events(response_id="r1")
    types = [e["type"] for e in evs]
    assert types[0] == "response.created" and types[-1] == "response.completed"
    added = next(e for e in evs if e["type"] == "response.output_item.added")
    assert added["item"]["type"] == "function_call" and added["item"]["name"] == "idle_do_nothing"
    done = evs[-1]["response"]["output"][0]
    assert done["name"] == "idle_do_nothing" and done["arguments"] == "{}"


def test_with_instructions_and_tools_merges_and_replaces():
    out = with_instructions_and_tools(REQ, instructions="PERSONA", extra_tools=[{"type": "function", "name": "roll", "parameters": {}}])
    assert out["instructions"] == "PERSONA"
    names = [t["name"] for t in out["tools"]]
    assert names == ["play_emotion", "roll"] and out is not REQ


# --- contract details verified against the installed speech-to-speech 1.0.0 source ---

def test_idle_events_call_id_is_client_supplied_with_call_prefix():
    """speech_to_speech/LLM/chat.py raises ChatItemError unless a function_call
    carries a client-supplied call_id starting with 'call_'."""
    evs = idle_response_events()
    item = evs[-1]["response"]["output"][0]
    assert item["call_id"].startswith("call_")
    added = next(e for e in evs if e["type"] == "response.output_item.added")
    assert added["item"]["call_id"] == item["call_id"]


def test_wants_stream_defaults_to_false_when_absent():
    """ResponsesApiModelHandler.warmup() omits `stream`, so absence means non-streaming."""
    assert wants_stream({"stream": True}) is True
    assert wants_stream({"stream": False}) is False
    assert wants_stream({}) is False


def test_idle_response_json_is_a_complete_response_object():
    """warmup() does a non-streaming responses.create; the openai SDK parses the body
    as a Response object, so it needs the envelope fields, not just output."""
    r = idle_response_json(model="familiar")
    assert r["object"] == "response" and r["status"] == "completed"
    assert r["model"] == "familiar" and isinstance(r["created_at"], int)
    assert r["id"].startswith("resp_")
    item = r["output"][0]
    assert item["type"] == "function_call" and item["name"] == "idle_do_nothing"
    assert item["call_id"].startswith("call_") and item["arguments"] == "{}"


def test_call_item_is_a_completed_function_call_with_json_arguments():
    item = call_item("play_emotion", {"emotion": "happy"})
    assert item["type"] == "function_call" and item["status"] == "completed"
    assert item["name"] == "play_emotion" and json.loads(item["arguments"]) == {"emotion": "happy"}
    assert item["call_id"].startswith("call_") and item["id"].startswith("fc_")
    assert call_item("idle_do_nothing")["arguments"] == "{}"


def test_idle_items_is_one_idle_call():
    (item,) = idle_items()
    assert item["name"] == "idle_do_nothing" and item["arguments"] == "{}"


def test_item_events_number_items_from_the_start_index():
    a, b = call_item("head_tracking", {"enabled": True}), call_item("play_emotion", {"emotion": "attentive"})
    evs = item_events([a, b], start_index=3)
    assert [e["type"] for e in evs] == [
        "response.output_item.added", "response.function_call_arguments.delta",
        "response.function_call_arguments.done", "response.output_item.done"] * 2
    assert [e["output_index"] for e in evs] == [3, 3, 3, 3, 4, 4, 4, 4]
    assert evs[0]["item"]["arguments"] == "" and evs[3]["item"] is a
    assert evs[1]["item_id"] == a["id"] and evs[1]["delta"] == a["arguments"]


def test_tool_call_events_wrap_items_in_a_response_envelope():
    items = [call_item("play_emotion", {"emotion": "happy"})]
    evs = tool_call_events(items, response_id="r9")
    assert evs[0]["type"] == "response.created" and evs[0]["response"]["id"] == "r9"
    assert evs[-1]["type"] == "response.completed" and evs[-1]["response"]["output"] == items


def test_empty_response_events_carry_no_items():
    evs = empty_response_events(response_id="r0")
    assert [e["type"] for e in evs] == ["response.created", "response.completed"]
    assert evs[-1]["response"]["output"] == []


# --- continuations: requests that are not a fresh utterance from the table ---

IMG = {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/AAA="}


def test_camera_frame_as_last_item_is_a_sight_continuation():
    req = {"input": [
        {"role": "user", "content": [{"type": "input_text", "text": "Rocky, what do you see?"}]},
        {"type": "function_call", "call_id": "call_1", "name": "camera", "arguments": "{\"question\":\"scene\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "{\"image_attached\": true}"},
        {"type": "message", "role": "user", "content": [IMG]},
    ]}
    assert continuation_kind(req) == "sight"


def test_camera_result_without_a_frame_is_still_sight_so_the_model_can_say_it_failed():
    req = {"input": [
        {"role": "user", "content": "Rocky, look"},
        {"type": "function_call", "call_id": "call_2", "name": "camera", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_2", "output": "{\"error\": \"Camera is disabled\"}"},
    ]}
    assert continuation_kind(req) == "sight"


def test_other_tool_results_are_echoes():
    req = {"input": [
        {"role": "user", "content": "the door creaks"},
        {"type": "function_call", "call_id": "call_3", "name": "idle_do_nothing", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_3", "output": "{\"status\": \"idle\"}"},
    ]}
    assert continuation_kind(req) == "echo"


def test_ordinary_utterances_and_the_greeting_prompt():
    assert continuation_kind({"input": [{"role": "user", "content": "Rocky, hello"}]}) is None
    assert continuation_kind({"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}, IMG]}]}) is None
    assert continuation_kind({"input": []}) is None
    greeting = {"input": [{"role": "user", "content": [{"type": "input_text", "text": GREETING_MARKER + " with a brief greeting"}]}]}
    assert continuation_kind(greeting) == "greeting"


def test_greeting_marker_in_a_bare_string_content_is_the_greeting():
    req = {"input": [{"role": "user", "content": GREETING_MARKER + " with a brief greeting"}]}
    assert continuation_kind(req) == "greeting"


def test_tool_result_whose_call_is_not_in_the_input_is_an_echo():
    """The client may have trimmed the function_call out of history; a tool result is
    still never an utterance, so it must not reach the gate."""
    req = {"input": [
        {"role": "user", "content": "the door creaks"},
        {"type": "function_call_output", "call_id": "call_gone", "output": "{\"status\": \"idle\"}"},
    ]}
    assert continuation_kind(req) == "echo"


def test_image_only_message_item_is_sight():
    req = {"input": [{"type": "message", "role": "user", "content": [IMG, dict(IMG)]}]}
    assert continuation_kind(req) == "sight"


def test_utterance_carrying_an_image_is_still_an_utterance():
    req = {"input": [{"type": "message", "role": "user",
                      "content": [IMG, {"type": "input_text", "text": "Rocky, is that a goblin?"}]}]}
    assert continuation_kind(req) is None


def test_malformed_or_missing_input_is_not_a_continuation():
    assert continuation_kind({}) is None
    assert continuation_kind({"input": None}) is None
    assert continuation_kind({"input": []}) is None
    assert continuation_kind({"input": ["Rocky, hello"]}) is None
    assert continuation_kind({"input": [{"role": "user", "content": "hi"}, None]}) is None
    # The Responses API also accepts a bare string as the whole input.
    assert continuation_kind({"input": "Rocky, hello"}) is None


# --- requests speech-to-speech makes that are not a line from the table ---

def _fc(name: str, call_id: str) -> list[dict]:
    return [{"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"},
            {"type": "function_call_output", "call_id": call_id, "output": "{}"}]


def test_the_tool_follow_up_prefetch_after_a_spoken_reply_is_an_echo():
    """speech-to-speech speculatively asks again after every tool output, with each output
    moved next to its call, so the request ends on the assistant's own reply."""
    req = {"input": [{"role": "user", "content": "Rocky, hello"}, *_fc("head_tracking", "call_a"),
                     *_fc("play_emotion", "call_b"),
                     {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Evening."}]}]}
    assert continuation_kind(req) == "echo"
    assert continuation_kind({"input": [*req["input"][:-1], {"role": "assistant", "content": "Evening."}]}) == "echo"


def test_a_line_after_the_assistant_reply_is_still_an_utterance():
    req = {"input": [{"role": "user", "content": "Rocky, hello"},
                     {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Evening."}]},
                     {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "And the door?"}]}]}
    assert continuation_kind(req) is None


def test_history_compaction_is_its_own_kind_even_when_it_quotes_the_greeting():
    transcript = ("Summarize the following conversation.  Return only the JSON object.\n\n"
                  "--- CONVERSATION START ---\nUser: " + GREETING_MARKER + " with a brief greeting.\n\n"
                  "Assistant: Evening.\n--- CONVERSATION END ---")
    req = {"input": [{"role": "system", "content": "You are a conversation memory compressor."},
                     {"role": "user", "content": transcript}]}
    assert continuation_kind(req) == "compaction"
