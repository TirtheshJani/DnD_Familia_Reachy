import json
from pathlib import Path

import pytest
import requests

from familiar_brain import indexing
from familiar_brain.app import create_app
from familiar_brain.motion import Choreographer
from familiar_brain.retrieval import Store
from familiar_brain.upstream import Upstream
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm_server import FakeLLM

PERSONA = """name: Rocky
aliases: [rocky]
tts_speaker: Aiden
manner: dry
gestures:
  attentive: play_emotion:attentive
  amused: play_emotion:happy
  alarmed: play_emotion:surprised
  thoughtful: play_emotion:thinking
  sad: play_emotion:sad
  scared: play_emotion:scared
  asleep: play_emotion:sleepy
  awake: play_emotion:welcoming
"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(indexing, "open_store", lambda cfg: Store(tmp_path / "k.db", embedder=FakeEmbedder()))
    persona = tmp_path / "persona.yaml"; persona.write_text(PERSONA)
    cfg = tmp_path / "brain.toml"
    cfg.write_text(f"""
[server]
host = "127.0.0.1"
port = 0
[upstream]
base_url = "{fake.base_url}"
generate_slot = 0
classify_slot = 1
classify_timeout_s = 1.0
[gate]
follow_up_window_s = 8
[paths]
persona = "{persona.as_posix()}"
sessions_dir = "{(tmp_path / 'sessions').as_posix()}"
[knowledge]
db_path = "{(tmp_path / 'k.db').as_posix()}"
notes_dir = "{(tmp_path / 'notes').as_posix()}"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
[motion]
min_gesture_gap_s = 0
reaction_gap_s = 0
glance_gap_s = 0
""")
    app = create_app(cfg)
    app.config["TESTING"] = True
    yield app.test_client(), fake, tmp_path, app
    fake.close()


def _calls(resp) -> list[tuple[str, dict]]:
    out = []
    for l in resp.get_data(as_text=True).splitlines():
        if l.startswith("data: "):
            ev = json.loads(l[6:]); item = ev.get("item") or {}
            if ev["type"] == "response.output_item.done" and item.get("type") == "function_call":
                out.append((item["name"], json.loads(item["arguments"])))
    return out


def _types(resp) -> list[str]:
    return [json.loads(l[6:])["type"] for l in resp.get_data(as_text=True).splitlines() if l.startswith("data: ")]


def _say(client, text):
    resp = client.post("/v1/responses", json={"model": "familiar", "stream": True, "tools": [],
                                              "input": [{"role": "user", "content": text}]})
    # Read the whole stream, as speech-to-speech does. The preface now comes first, so a
    # response left unread stops before the upstream call and before the log line.
    resp.get_data()
    return resp


def _chat_calls(fake) -> list[dict]:
    return [r["body"] for r in fake.requests if r["path"].endswith("/chat/completions")]


def test_addressed_turn_is_prefaced_by_tracking_then_a_gesture_then_speech(env):
    client, fake, _, _ = env
    fake.chat_replies.append("thoughtful")      # the mood call; classify is skipped on a named line
    resp = _say(client, "Rocky, what do you make of him?")
    calls = _calls(resp)
    assert calls == [("head_tracking", {"enabled": True}), ("play_emotion", {"emotion": "thinking"})]
    types = _types(resp)
    assert types.index("response.output_item.done") < types.index("response.output_text.delta")


def test_ambient_outburst_gets_a_rule_gesture_and_no_speech_or_idle(env):
    client, fake, _, _ = env
    fake.chat_replies.append("AMBIENT")         # the gate; "Natural twenty!" hits a rule, so no reaction call
    resp = _say(client, "Natural twenty!")
    assert _calls(resp) == [("play_emotion", {"emotion": "happy"})]
    assert "response.output_text.delta" not in _types(resp)


def test_narration_twist_gets_a_surprised_reaction_from_the_reaction_call(env):
    client, fake, _, _ = env
    fake.chat_replies.append("IGNORE")          # the gate: narration is IGNORE (deviation E8)
    fake.chat_replies.append("alarmed")         # the reaction call
    resp = _say(client, "The priest pulls back his hood. It is the king, and he has been dead a year.")
    assert _calls(resp) == [("play_emotion", {"emotion": "surprised"})]
    assert "response.output_text.delta" not in _types(resp)
    assert "not spoken to it" in fake.requests[-1]["body"]["messages"][0]["content"]


def test_sad_narration_gets_a_sad_reaction(env):
    client, fake, _, _ = env
    fake.chat_replies.append("IGNORE")
    fake.chat_replies.append("sad")
    resp = _say(client, "She folds the letter and does not read the rest aloud.")
    assert _calls(resp) == [("play_emotion", {"emotion": "sad"})]


def test_plain_narration_answered_none_still_idles(env):
    client, fake, _, _ = env
    fake.chat_replies.append("IGNORE")
    fake.chat_replies.append("none")
    assert _calls(_say(client, "You walk for an hour along the river road.")) == [("idle_do_nothing", {})]


def test_short_ignored_line_makes_no_reaction_call(env):
    client, fake, _, _ = env
    fake.chat_replies.append("IGNORE")
    before = len(fake.requests)
    assert _calls(_say(client, "the door creaks")) == [("idle_do_nothing", {})]
    assert len(fake.requests) - before == 1     # the gate only


def test_tracking_is_switched_off_once_the_window_and_linger_have_passed(env):
    client, fake, _, app = env
    fake.chat_replies.append("attentive")       # the mood call for the named line
    _say(client, "Rocky, hello")
    bs = app.extensions["brain_state"]
    bs.last_named_at -= 60; bs.last_spoke_at = (bs.last_spoke_at or 0) - 60
    fake.chat_replies.append("IGNORE")
    assert ("head_tracking", {"enabled": False}) in _calls(_say(client, "moving on"))


def test_mute_and_unmute_cue_sleep_and_wake_gestures_and_muted_never_reacts(env):
    client, fake, _, _ = env
    client.post("/api/mute")
    assert _calls(_say(client, "Rocky, hello")) == [("play_emotion", {"emotion": "sleepy"})]
    before = len(fake.requests)
    assert _calls(_say(client, "The priest pulls back his hood and it is the king.")) == [("idle_do_nothing", {})]
    assert len(fake.requests) == before         # muted: no gate call, no reaction call
    client.post("/api/unmute")
    fake.chat_replies.append("IGNORE")
    assert _calls(_say(client, "table talk")) == [("play_emotion", {"emotion": "welcoming"})]


def test_mode_api_and_phrase_change_the_length_block_sent_upstream(env):
    client, fake, _, _ = env
    _say(client, "Rocky, think with me about the cult")
    assert "up to six sentences" in fake.requests[-1]["body"]["instructions"]
    assert client.get("/api/state").get_json()["mode"] == "thinking"
    assert client.post("/api/mode", json={"mode": "familiar"}).status_code == 200
    _say(client, "Rocky, and the docks?")
    assert "two spoken sentences" in fake.requests[-1]["body"]["instructions"]
    assert client.post("/api/mode", json={"mode": "loud"}).status_code == 400


# --- review notes and gaps found reading the code ---

def test_mood_call_never_offers_the_sleep_and_wake_cues(env):
    """asleep and awake are the mute and unmute cues, not moods to answer in."""
    client, fake, _, _ = env
    fake.chat_replies.append("thoughtful")      # the mood call; a named line skips classify
    _say(client, "Rocky, what do you make of him?")
    [mood_call] = _chat_calls(fake)
    system = mood_call["messages"][0]["content"]
    assert "thoughtful" in system and "attentive" in system
    assert "asleep" not in system and "awake" not in system


def test_tracking_off_goes_out_ahead_of_the_mute_cue(env):
    client, fake, _, _ = env
    fake.chat_replies.append("attentive")       # the mood call for the named line
    assert ("head_tracking", {"enabled": True}) in _calls(_say(client, "Rocky, hello"))
    assert client.get("/api/state").get_json()["tracking"] is True
    client.post("/api/mute")
    assert _calls(_say(client, "moving on")) == [("head_tracking", {"enabled": False}),
                                                  ("play_emotion", {"emotion": "sleepy"})]
    assert client.get("/api/state").get_json()["tracking"] is False


def test_a_failing_gesture_never_costs_the_reply(env, monkeypatch):
    client, _, _, _ = env

    def broken(self, text, now):
        raise RuntimeError("choreographer bug")

    monkeypatch.setattr(Choreographer, "for_addressed", broken)
    resp = _say(client, "Rocky, what do you make of him?")
    assert resp.status_code == 200 and "response.output_text.delta" in _types(resp)
    assert _calls(resp) == [("head_tracking", {"enabled": True})]   # what was chosen before it failed


def test_upstream_failure_after_the_preface_still_degrades_to_silence(env, monkeypatch):
    client, fake, _, _ = env

    def down(self, req):
        raise requests.ConnectionError("llama-server is down")
        yield  # a generator, like the real Upstream.stream

    monkeypatch.setattr(Upstream, "stream", down)
    fake.chat_replies.append("thoughtful")      # the mood call
    resp = _say(client, "Rocky, what do you make of him?")
    assert "response.output_text.delta" not in _types(resp)
    assert _calls(resp) == [("head_tracking", {"enabled": True}), ("play_emotion", {"emotion": "thinking"}),
                            ("idle_do_nothing", {})]


def test_non_streaming_turn_uses_the_mode_aware_prompt(env):
    client, fake, _, _ = env
    client.post("/api/mode", json={"mode": "thinking"})
    resp = client.post("/v1/responses", json={"model": "familiar",
                                              "input": [{"role": "user", "content": "Rocky, and the docks?"}]})
    assert resp.headers["Content-Type"].startswith("application/json")
    sent = fake.requests[-1]["body"]
    assert sent["stream"] is False and "up to six sentences" in sent["instructions"]
    assert _chat_calls(fake) == []              # no preface on this path, so no mood call


def test_mode_set_through_the_api_times_out(env):
    """POST /api/mode stamps the mode on the clock update_mode compares against, and
    /api/state applies the timeout rather than showing a mode the next turn would end."""
    client, _, _, app = env
    assert client.post("/api/mode", json={"mode": "thinking"}).get_json() == {"mode": "thinking"}
    bs = app.extensions["brain_state"]
    assert bs.mode_set_at is not None
    bs.mode_set_at -= 301                       # mode_timeout_s defaults to 300
    assert client.get("/api/state").get_json()["mode"] == "familiar"
    assert client.post("/api/mode", json=["thinking"]).status_code == 400
