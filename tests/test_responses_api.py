import json
from familiar_brain.responses_api import (
    latest_user_text, function_call_outputs, sse, idle_response_events, with_instructions_and_tools,
    idle_response_json, wants_stream,
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
