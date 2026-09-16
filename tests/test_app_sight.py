import json
import logging
from pathlib import Path

from tests.test_app_expression import env, _calls, _types, _say  # noqa: F401  (fixture reuse)

IMG = {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/AAA="}


def _camera_round_trip(text, output):
    return {"model": "familiar", "stream": True, "tools": [], "input": [
        {"role": "user", "content": [{"type": "input_text", "text": text}]},
        {"type": "function_call", "call_id": "call_c1", "name": "camera", "arguments": "{\"question\":\"scene\"}"},
        {"type": "function_call_output", "call_id": "call_c1", "output": output},
        {"type": "message", "role": "user", "content": [IMG]},
    ]}


def _idle_echo(stream: bool) -> dict:
    return {"model": "familiar", "stream": stream, "input": [
        {"role": "user", "content": "the door creaks"},
        {"type": "function_call", "call_id": "call_i1", "name": "idle_do_nothing", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_i1", "output": "{\"status\": \"idle\"}"},
    ]}


GREETING = {"model": "familiar", "stream": True, "input": [
    {"type": "message", "role": "user", "content": [{"type": "input_text",
     "text": "Start the conversation now with a brief, spontaneous greeting in character."}]}]}


def _session_log(tmp: Path) -> str:
    return next((tmp / "sessions").glob("*.jsonl")).read_text()


def test_camera_frame_reaches_upstream_and_is_never_logged(env):
    client, fake, tmp, _ = env
    client.post("/api/session/start")
    _say(client, "Rocky, what do you see?")
    fake.responses_events.append([
        {"type": "response.output_text.delta", "delta": "Four mortals and a great many dice."},
        {"type": "response.completed", "response": {"id": "r", "output": []}},
    ])
    resp = client.post("/v1/responses", json=_camera_round_trip("Rocky, what do you see?", "{\"image_attached\": true}"))
    assert "response.output_text.delta" in _types(resp)
    sent = fake.requests[-1]["body"]["input"]
    assert sent[-1]["content"][0]["type"] == "input_image" and sent[-1]["content"][0]["image_url"].startswith("data:image/jpeg")
    log = _session_log(tmp)
    assert "data:image" not in log and log.count("\n") == 1       # only the spoken question was logged


def test_camera_continuation_is_silent_while_muted(env):
    client, fake, _, _ = env
    _say(client, "Rocky, what do you see?")
    client.post("/api/mute")
    resp = client.post("/v1/responses", json=_camera_round_trip("Rocky, what do you see?", "{\"image_attached\": true}"))
    assert "response.output_text.delta" not in _types(resp)
    assert ("play_emotion", {"emotion": "sleepy"}) in _calls(resp)


def test_tool_result_echo_gets_an_empty_response_and_no_log_line(env):
    client, fake, tmp, _ = env
    client.post("/api/session/start")
    fake.chat_replies.append("IGNORE")
    _say(client, "the door creaks")
    resp = client.post("/v1/responses", json=_idle_echo(stream=True))
    assert _calls(resp) == [] and _types(resp) == ["response.created", "response.completed"]
    assert _session_log(tmp).count("\n") == 1


def test_greeting_prompt_is_answered_with_a_wake_gesture_and_no_open_window(env):
    client, fake, _, app = env
    fake.responses_events.append([
        {"type": "response.output_text.delta", "delta": "Evening."},
        {"type": "response.completed", "response": {"id": "r", "output": []}},
    ])
    resp = client.post("/v1/responses", json=GREETING)
    assert ("play_emotion", {"emotion": "welcoming"}) in _calls(resp) and "response.output_text.delta" in _types(resp)
    assert app.extensions["brain_state"].last_named_at is None


# --- review notes and gaps found reading the code ---

def test_non_streaming_echo_gets_an_empty_json_response(env):
    client, fake, tmp, _ = env
    client.post("/api/session/start")
    before = len(fake.requests)
    resp = client.post("/v1/responses", json=_idle_echo(stream=False))
    assert resp.status_code == 200 and resp.headers["Content-Type"].startswith("application/json")
    body = resp.get_json()
    assert body["object"] == "response" and body["status"] == "completed" and body["output"] == []
    assert len(fake.requests) == before         # no gate, no mood call, no upstream
    assert _session_log(tmp) == ""


def test_continuations_never_touch_the_session_log_or_the_follow_up_state(env):
    """Sight and greeting, streamed or not: no log line, no remember, no mark_spoke."""
    client, fake, tmp, app = env
    client.post("/api/session/start")
    _say(client, "Rocky, what do you see?")
    bs = app.extensions["brain_state"]
    snapshot = lambda: (list(bs.recent_segments), bs.last_spoke_at, bs.last_named_at, bs.follow_ups)
    before, log_before = snapshot(), _session_log(tmp)

    sight = _camera_round_trip("Rocky, what do you see?", "{\"image_attached\": true}")
    assert "response.output_text.delta" in _types(client.post("/v1/responses", json=sight))
    resp = client.post("/v1/responses", json={**sight, "stream": False})
    assert resp.headers["Content-Type"].startswith("application/json")
    sent = fake.requests[-1]["body"]
    assert sent["stream"] is False and sent["input"][-1]["content"][0]["type"] == "input_image"
    assert "Rocky" in sent["instructions"]
    assert "response.output_text.delta" in _types(client.post("/v1/responses", json=GREETING))
    client.post("/v1/responses", json={**GREETING, "stream": False})

    assert snapshot() == before
    assert _session_log(tmp) == log_before and log_before.count("\n") == 1


def test_muted_brain_answers_no_frame_and_no_greeting_on_either_path(env):
    client, fake, _, app = env
    client.post("/api/mute")
    before = len(fake.requests)
    sight = _camera_round_trip("Rocky, what do you see?", "{\"image_attached\": true}")
    for req in (sight, GREETING):
        body = client.post("/v1/responses", json={**req, "stream": False}).get_json()
        assert [i["name"] for i in body["output"]] == ["idle_do_nothing"]
    resp = client.post("/v1/responses", json=GREETING)
    assert "response.output_text.delta" not in _types(resp)
    assert _calls(resp) == [("play_emotion", {"emotion": "sleepy"})]   # the mute cue, no wake cue
    assert len(fake.requests) == before         # nothing reached llama-server
    assert app.extensions["brain_state"].tracking is False


def test_camera_frame_never_reaches_a_log_record_and_timing_names_the_continuation(env, caplog):
    """The frame must not reach brain.log either. The timing lines say SIGHT, ECHO and
    GREETING, so gate-report, which counts ADDRESSED lines only, never scores them."""
    client, _, _, _ = env
    with caplog.at_level(logging.DEBUG):
        sight = _camera_round_trip("Rocky, what do you see?", "{\"image_attached\": true}")
        client.post("/v1/responses", json=sight).get_data()
        client.post("/v1/responses", json={**sight, "stream": False})
        client.post("/v1/responses", json=_idle_echo(stream=True)).get_data()
        client.post("/v1/responses", json=GREETING).get_data()
    messages = [r.getMessage() for r in caplog.records]
    assert not any("/9j/AAA" in m or "data:image" in m for m in messages)
    decisions = [json.loads(m)["decision"] for m in messages if m.startswith("{") and '"decision"' in m]
    assert decisions == ["SIGHT", "SIGHT", "ECHO", "GREETING"]
