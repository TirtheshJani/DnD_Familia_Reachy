from __future__ import annotations

import json

from familiar_brain.tools import initiative as initiative_tool
from scripts import run_s2s
from tests.test_app_expression import env, _calls, _say, _types  # noqa: F401
from tests.test_app_sight import _camera_round_trip


def test_camera_consent_toggle_blocks_tool_and_sight_when_closed(env):  # noqa: F811
    client, fake, _, _ = env
    # Close Eyes explicitly
    r = client.post("/api/camera", json={"enabled": False})
    assert r.status_code == 200 and r.get_json()["camera_enabled"] is False

    # Addressed turn with camera tool offered must strip camera from upstream tools
    fake.responses_events.append([
        {"type": "response.output_text.delta", "delta": "My eyes are closed right now."},
        {"type": "response.completed", "response": {"id": "r1", "output": []}},
    ])
    req = {
        "model": "familiar",
        "stream": True,
        "tools": [
            {"type": "function", "name": "camera", "parameters": {"type": "object", "properties": {}}},
            {"type": "function", "name": "play_emotion", "parameters": {"type": "object", "properties": {}}},
        ],
        "input": [{"role": "user", "content": "Rocky, what do you see?"}],
    }
    resp_ask = client.post("/v1/responses", json=req)
    _ = resp_ask.get_data(as_text=True)
    sent_tools = [t["name"] for t in fake.requests[-1]["body"].get("tools", [])]
    assert "camera" not in sent_tools
    assert "eyes (camera) are currently closed" in fake.requests[-1]["body"]["instructions"]

    # Sight continuation while Eyes Closed must be ignored (no image forwarded upstream)
    before_count = len(fake.requests)
    resp = client.post("/v1/responses", json=_camera_round_trip("Rocky, what do you see?", '{"image_attached": true}'))
    assert "response.output_text.delta" not in _types(resp)
    assert len(fake.requests) == before_count

    # Open Eyes -> camera tool returns
    r2 = client.post("/api/camera", json={"enabled": True})
    assert r2.status_code == 200 and r2.get_json()["camera_enabled"] is True


def test_phase2_companion_endpoints_behavior_initiative_roll_memory_chat_voice_sse(env, monkeypatch):  # noqa: F811
    client, fake, tmp, app = env
    initiative_tool.TRACKER.clear()

    # 1. Behavior dials + voice shortcut
    r = client.post("/api/behavior", json={"humor": 80, "honesty": 95, "bravery": 70})
    assert r.status_code == 200
    assert r.get_json()["humor"] == 80
    assert client.get("/api/behavior").get_json()["honesty"] == 95

    # Spoken voice command updates dial immediately on addressed turn
    fake.responses_events.append([
        {"type": "response.output_text.delta", "delta": "Humor set to fifty percent."},
        {"type": "response.completed", "response": {"id": "r2", "output": []}},
    ])
    _say(client, "Rocky, set humor to 50 percent")
    assert client.get("/api/behavior").get_json()["humor"] == 50

    # 2. Initiative API
    r_add = client.post("/api/initiative", json={"action": "add", "name": "Brokk", "value": 18})
    assert r_add.status_code == 200
    client.post("/api/initiative", json={"action": "add", "name": "Krag", "value": 14})
    r_next = client.post("/api/initiative", json={"action": "next"})
    assert r_next.get_json()["current"] == "Brokk"
    assert len(client.get("/api/initiative").get_json()["order"]) == 2
    client.post("/api/initiative", json={"action": "clear"})
    assert client.get("/api/initiative").get_json()["order"] == []

    # 3. Quick dice roll API
    r_roll = client.post("/api/roll", json={"expression": "2d6+3", "advantage": "none"})
    assert r_roll.status_code == 200
    assert "total" in r_roll.get_json()["result"].lower()

    # 4. Campaign memory viewer/editor
    r_put = client.put("/api/memory", json={"campaign_memory": "## Story so far\n- Brokk met the Jade Council.\n"})
    assert r_put.status_code == 200 and r_put.get_json()["indexed_chunks"] >= 1
    r_mem = client.get("/api/memory")
    assert "Jade Council" in r_mem.get_json()["campaign_memory"]

    # 5. Silent whisper chat (uses full streaming tool loop)
    fake.responses_events.append([
        {"type": "response.output_text.delta", "delta": "The docks smell of brine."},
        {"type": "response.completed", "response": {"id": "resp_chat", "output": []}},
    ])
    r_chat = client.post("/api/chat", json={"text": "Tell me about the docks."})
    assert r_chat.status_code == 200
    assert "brine" in r_chat.get_json()["reply"]

    # 6. Live Kokoro voice switcher + run_s2s live pickup
    r_voice = client.post("/api/voice", json={"voice": "bm_george"})
    assert r_voice.status_code == 200 and r_voice.get_json()["voice"] == "bm_george"
    voice_file = tmp / "voice.json"
    assert voice_file.exists()
    monkeypatch.setenv("FAMILIAR_VOICE_FILE", str(voice_file))

    class FakeKokoro:
        voice = "bm_fable"
        _initial_voice = "bm_fable"

        def _process_kokoro(self, text: str, language_code=None):
            yield (self.voice, language_code)

    run_s2s.patch_kokoro_voice_guard(FakeKokoro)
    out = list(FakeKokoro()._process_kokoro("Hello."))
    assert out == [("bm_george", "b")]

    # 7. Robot heartbeat updates /api/state immediately
    r_hb = client.post("/api/robot/heartbeat", json={"backend_connected": True, "host": "10.0.0.100", "app": "reachy_familiar"})
    assert r_hb.status_code == 200
    st = client.get("/api/state").get_json()
    assert st["robot"]["backend_connected"] is True
    assert "reachy_familiar" in st["robot"]["detail"]
    assert st["voice"] == "bm_george"

    # 8. One-shot SSE stream returns valid state event
    r_sse = client.get("/api/events?once=1")
    assert r_sse.status_code == 200
    raw_sse = r_sse.get_data(as_text=True)
    assert "event: state" in raw_sse
    data_line = [ln for ln in raw_sse.splitlines() if ln.startswith("data:")][0]
    parsed = json.loads(data_line[5:].strip())
    assert parsed["behavior"]["humor"] == 50
