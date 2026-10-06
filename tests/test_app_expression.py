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
  cheer: play_emotion:success
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


TRACK_ON, TRACK_OFF = ("head_tracking", {"enabled": True}), ("head_tracking", {"enabled": False})


def _wake(client, fake):
    """One short ignored line, so the tracker is already on and the next turn carries only
    its own gesture. Three words make no reaction call."""
    fake.chat_replies.append("IGNORE")
    assert _calls(_say(client, "dice please")) == [TRACK_ON]


def test_addressed_turn_is_prefaced_by_tracking_then_a_gesture_then_speech(env):
    client, fake, _, _ = env
    resp = _say(client, "Rocky, what do you make of him?")
    calls = _calls(resp)
    assert calls == [("head_tracking", {"enabled": True}), ("play_emotion", {"emotion": "thinking"})]
    assert _chat_calls(fake) == []              # a named line skips classify; a rule hit skips the mood call
    types = _types(resp)
    assert types.index("response.output_item.done") < types.index("response.output_text.delta")


def test_ambient_outburst_gets_a_rule_gesture_and_no_speech_or_idle(env):
    client, fake, _, _ = env
    fake.chat_replies.append("AMBIENT")         # the gate; "Natural twenty!" hits a rule, so no reaction call
    resp = _say(client, "Natural twenty!")
    assert _calls(resp) == [TRACK_ON, ("play_emotion", {"emotion": "happy"})]
    assert "response.output_text.delta" not in _types(resp)


def test_narration_twist_gets_a_surprised_reaction_from_the_reaction_call(env):
    client, fake, _, _ = env
    fake.chat_replies.append("IGNORE")          # the gate: narration is IGNORE (deviation E8)
    fake.chat_replies.append("alarmed")         # the reaction call
    resp = _say(client, "The priest pulls back his hood. It is the king, and he has been dead a year.")
    assert _calls(resp) == [TRACK_ON, ("play_emotion", {"emotion": "surprised"})]
    assert "response.output_text.delta" not in _types(resp)
    assert "not spoken to it" in fake.requests[-1]["body"]["messages"][0]["content"]


def test_sad_narration_gets_a_sad_reaction(env):
    client, fake, _, _ = env
    fake.chat_replies.append("IGNORE")
    fake.chat_replies.append("sad")
    resp = _say(client, "She folds the letter and does not read the rest aloud.")
    assert _calls(resp) == [TRACK_ON, ("play_emotion", {"emotion": "sad"})]


def test_plain_narration_answered_none_still_idles(env):
    client, fake, _, _ = env
    _wake(client, fake)
    fake.chat_replies.append("IGNORE")
    fake.chat_replies.append("none")
    assert _calls(_say(client, "You walk for an hour along the river road.")) == [("idle_do_nothing", {})]


def test_short_ignored_line_makes_no_reaction_call(env):
    client, fake, _, _ = env
    _wake(client, fake)
    fake.chat_replies.append("IGNORE")
    before = len(fake.requests)
    assert _calls(_say(client, "the door creaks")) == [("idle_do_nothing", {})]
    assert len(fake.requests) - before == 1     # the gate only


def test_the_first_line_switches_tracking_on_even_when_nobody_spoke_to_rocky(env):
    """The tracker is on from the first thing anyone says, not from the first naming."""
    client, fake, _, app = env
    fake.chat_replies.append("IGNORE")
    resp = _say(client, "the door creaks")
    assert _calls(resp) == [TRACK_ON] and "response.output_text.delta" not in _types(resp)
    assert app.extensions["brain_state"].tracking is True


def test_tracking_stays_on_across_every_kind_of_turn_with_no_repeated_toggle(env):
    """Once on, nothing switches it off but a mute, not even the follow-up window closing,
    and a repeated head_tracking(true) would reset the app's speaking handoff."""
    client, fake, _, app = env
    _wake(client, fake)
    assert TRACK_ON not in _calls(_say(client, "Rocky, hello"))
    bs = app.extensions["brain_state"]
    bs.last_named_at -= 60; bs.last_spoke_at = (bs.last_spoke_at or 0) - 60
    fake.chat_replies.append("AMBIENT")
    assert _calls(_say(client, "Natural twenty!")) == [("play_emotion", {"emotion": "happy"})]
    fake.chat_replies.append("IGNORE")
    assert _calls(_say(client, "moving on")) == [("idle_do_nothing", {})]
    assert bs.tracking is True


def test_mute_and_unmute_cue_sleep_and_wake_gestures_and_muted_never_reacts(env):
    client, fake, _, _ = env
    client.post("/api/mute")
    assert _calls(_say(client, "Rocky, hello")) == [("play_emotion", {"emotion": "sleepy"})]
    before = len(fake.requests)
    assert _calls(_say(client, "The priest pulls back his hood and it is the king.")) == [("idle_do_nothing", {})]
    assert len(fake.requests) == before         # muted: no gate call, no reaction call
    client.post("/api/unmute")
    fake.chat_replies.append("IGNORE")
    assert _calls(_say(client, "table talk")) == [TRACK_ON, ("play_emotion", {"emotion": "welcoming"})]


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

def test_a_plain_addressed_reply_makes_no_mood_call_and_plays_no_emotion(env):
    """Live, the mood call timed out on 11 of 11 addressed turns and fell back to attentive,
    whose 4.28 s move played over the start of every reply."""
    client, fake, _, _ = env
    resp = _say(client, "Rocky, and the docks?")
    assert _calls(resp) == [("head_tracking", {"enabled": True})]
    assert _chat_calls(fake) == [] and "response.output_text.delta" in _types(resp)


def test_tracking_off_goes_out_ahead_of_the_mute_cue(env):
    client, fake, _, _ = env
    assert ("head_tracking", {"enabled": True}) in _calls(_say(client, "Rocky, hello"))
    assert client.get("/api/state").get_json()["tracking"] is True
    client.post("/api/mute")
    assert _calls(_say(client, "moving on")) == [TRACK_OFF, ("play_emotion", {"emotion": "sleepy"})]
    assert client.get("/api/state").get_json()["tracking"] is False
    assert _calls(_say(client, "Rocky, wake up")) == [("idle_do_nothing", {})]   # asleep: no toggle again


def test_unmute_switches_tracking_back_on_ahead_of_the_wake_cue(env):
    client, fake, _, _ = env
    _wake(client, fake)
    client.post("/api/mute")
    assert _calls(_say(client, "moving on")) == [TRACK_OFF, ("play_emotion", {"emotion": "sleepy"})]
    client.post("/api/unmute")
    fake.chat_replies.append("IGNORE")
    assert _calls(_say(client, "back again")) == [TRACK_ON, ("play_emotion", {"emotion": "welcoming"})]
    assert client.get("/api/state").get_json()["tracking"] is True


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
    resp = _say(client, "Rocky, what do you make of him?")  # a rule hit: no mood call
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


# --- 2026-10-02: more expression from cheap rules (deviation P15) ---

def test_an_ambient_burst_with_no_rule_and_no_reaction_still_looks_surprised(env):
    """Live the reaction call timed out on some lines, and the gate says AMBIENT only for
    a burst of excitement or dismay, so Rocky used to sit still through the loudest
    moments at the table. An explicit none plays nothing (deviation P16)."""
    client, fake, _, _ = env
    _wake(client, fake)
    fake.chat_replies.append("AMBIENT")
    fake.chat_replies.append("banana")          # the reaction call, asked although the line is short; unusable
    before = len(fake.requests)
    resp = _say(client, "ugh, seriously")
    assert _calls(resp) == [("play_emotion", {"emotion": "surprised"})]
    assert "response.output_text.delta" not in _types(resp)
    assert len(fake.requests) - before == 2     # the gate and the reaction call


def test_an_ambient_burst_plays_what_the_reaction_call_chose(env):
    client, fake, _, _ = env
    _wake(client, fake)
    fake.chat_replies.append("AMBIENT")
    fake.chat_replies.append("cheer")
    assert _calls(_say(client, "the whole table is on its feet")) == [("play_emotion", {"emotion": "success"})]


def test_a_victory_shouted_at_the_table_plays_success_without_a_reaction_call(env):
    client, fake, _, _ = env
    _wake(client, fake)
    fake.chat_replies.append("IGNORE")
    before = len(fake.requests)
    assert _calls(_say(client, "We did it! The dragon is finally dead, everyone!")) == [
        ("play_emotion", {"emotion": "success"})]
    assert len(fake.requests) - before == 1     # the gate only: a rule hit skips the call


def test_a_greeting_to_rocky_waves_but_a_request_after_hey_does_not(env):
    client, fake, _, _ = env
    _wake(client, fake)
    resp = _say(client, "Good evening, Rocky!")
    assert _calls(resp) == [("play_emotion", {"emotion": "welcoming"})]
    assert "response.output_text.delta" in _types(resp)
    assert _calls(_say(client, "Hey Rocky, roll me a d20")) == []


def test_an_ambient_line_the_reaction_call_answers_none_still_idles(env):
    """The classifier labels some narration AMBIENT; the reaction call's none is right."""
    client, fake, _, _ = env
    _wake(client, fake)
    fake.chat_replies.append("AMBIENT")
    fake.chat_replies.append("none")
    resp = _say(client, "You descend into the cistern. Water drips somewhere in the dark.")
    assert _calls(resp) == [("idle_do_nothing", {})]
