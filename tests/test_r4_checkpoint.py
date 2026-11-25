from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import pytest
import requests

from familiar_brain import indexing, learning
from familiar_brain.app import create_app
from familiar_brain.campaign import CAMPAIGN_FILE, campaign_due, folded, foldable, refresh_campaign
from familiar_brain.config import load_config
from familiar_brain.embeddings import DeterministicEmbedder, cosine
from familiar_brain.gate import Decision, Gate
from familiar_brain.live_index import BATCH_AGE_S
from familiar_brain.llm import LLMClient
from familiar_brain.motion import sleep_transition, table_mood
from familiar_brain.retrieval import Chunk, Store
from familiar_brain.session import SessionLog
from familiar_brain.state import BrainState
from familiar_brain.summariser import (
    Paused,
    Purged,
    apply_retention,
    checkpoint_session,
    purge_session,
    summarise_session,
)
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm_server import FakeLLM


def _make_app(tmp_path: Path, *, fake: FakeLLM, monkeypatch=None, extra_toml: str = "", use_deterministic: bool = False):
    if not use_deterministic and monkeypatch is not None:
        store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
        monkeypatch.setattr(indexing, "open_store", lambda cfg: store)
    else:
        store = None
    persona = tmp_path / "persona.yaml"
    persona.write_text(
        "name: Rocky\naliases: [rocky, rocki, rockie]\ntts_speaker: Aiden\nmanner: dry\n"
        "gestures:\n"
        "  attentive: play_emotion:attentive\n"
        "  amused: play_emotion:happy\n"
        "  alarmed: play_emotion:surprised\n"
        "  thoughtful: play_emotion:thinking\n"
        "  sad: play_emotion:sad\n"
        "  scared: play_emotion:scared\n"
        "  asleep: play_emotion:sleepy\n"
        "  awake: play_emotion:welcoming\n"
        "  cheer: play_emotion:success\n",
        encoding="utf-8",
    )
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "sessions").mkdir(exist_ok=True)
    embedder_line = 'embedder = "deterministic"\n' if use_deterministic else ""
    cfg = tmp_path / "brain.toml"
    cfg.write_text(
        f"""
[server]
host = "127.0.0.1"
port = 0
[upstream]
base_url = "{fake.base_url}"
generate_slot = 0
classify_slot = 1
classify_timeout_s = 1.0
classify_cooldown_s = 30.0
[gate]
follow_up_window_s = 8
[paths]
persona = "{persona.as_posix()}"
sessions_dir = "{(tmp_path / 'sessions').as_posix()}"
players = "{(tmp_path / 'players.txt').as_posix()}"
[knowledge]
db_path = "{(tmp_path / 'k.db').as_posix()}"
notes_dir = "{(tmp_path / 'notes').as_posix()}"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
{embedder_line}{extra_toml}
""",
        encoding="utf-8",
    )
    app = create_app(cfg)
    app.config["TESTING"] = True
    return app, store


def _turn(client, text: str, *, stream: bool = True):
    resp = client.post(
        "/v1/responses",
        json={"model": "familiar", "stream": stream, "tools": [], "input": [{"role": "user", "content": text}]},
    )
    resp.get_data()
    return resp


def _calls(resp) -> list[tuple[str, dict]]:
    out = []
    for line in resp.get_data(as_text=True).splitlines():
        if line.startswith("data: "):
            ev = json.loads(line[6:])
            item = ev.get("item") or {}
            if ev["type"] == "response.output_item.done" and item.get("type") == "function_call":
                out.append((item["name"], json.loads(item["arguments"])))
    return out


# --- R2.3: Defaults, slot-1 cooldown, and preserved accuracy ---


def test_default_brain_toml_disables_slot1_calls_and_sets_cooldown():
    cfg = load_config(Path("config/brain.toml"))
    assert cfg.gate.llm_classifier is False
    assert cfg.motion.llm_reaction is False
    assert cfg.upstream.classify_cooldown_s == 30
    assert cfg.knowledge.embedder == "bge"


def test_llm_client_slot1_cooldown_prevents_repeated_timeouts(monkeypatch):
    clock = {"t": 100.0}
    attempts = []

    def fake_post(url, *a, **kw):
        attempts.append(kw["json"]["messages"][0]["content"])
        raise requests.ReadTimeout("slot 1 busy")

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient(
        base_url="http://127.0.0.1:9/v1",
        classify_slot=1,
        classify_timeout_s=0.5,
        classify_cooldown_s=30.0,
        clock=lambda: clock["t"],
    )
    with pytest.raises(requests.ReadTimeout):
        client.classify(["the innkeeper pours ale"], persona_name="Rocky")
    assert len(attempts) == 1

    # Immediate reaction mood call and subsequent classify calls skip HTTP during cooldown
    assert client.mood("the innkeeper pours ale", moods=("amused", "none")) is None
    assert client.classify(["another line at the table"], persona_name="Rocky") == Decision.IGNORE
    assert len(attempts) == 1

    # After 30s, slot 1 is attempted again
    clock["t"] += 30.0
    with pytest.raises(requests.ReadTimeout):
        client.mood("the innkeeper pours ale", moods=("amused", "none"))
    assert len(attempts) == 2


def test_live_uninvited_rocky_and_table_rules_remain_accurate_with_classifier_off():
    live_row = json.loads((Path(__file__).parent / "fixtures" / "live_uninvited_rocky.jsonl").read_text(encoding="utf-8"))
    st = BrainState()
    g = Gate(state=st, name="Rocky", aliases=("rocky", "rocki", "rockie"), follow_up_window_s=8.0, classifier=None)
    assert g.decide([live_row["text"]]) == Decision.IGNORE
    assert table_mood("Natural twenty!") == "amused"


# --- R4: Mid-session checkpointing & campaign memory ---


def test_checkpoint_session_writes_checkpoint_without_touching_live_chunks_or_sentinel(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    sdir = tmp_path / "sessions"
    log = SessionLog(sdir)
    sid = log.start()
    log.append(text="The dockmaster hides a silver key.", decision="IGNORE", spoke=False)
    log.append(text="Bram owes ten gold.", decision="ADDRESSED", spoke=False, tag="deliberate")
    # Simulate live indexer chunks under <sid>.jsonl
    store.add_chunks([Chunk("sessions", f"{sid}.jsonl", "Session transcript", "The dockmaster hides a silver key.", {"session": sid, "piece": 0}, 1.0)])

    class FakeSummLLM:
        def complete(self, messages, max_tokens=1200):
            return "## Scenes\n- The docks\n## NPCs met\n- Dockmaster\n"

    (tmp_path / "players.txt").write_text("James\n", encoding="utf-8")
    snap = log.snapshot_text()
    ckpt = checkpoint_session(log.path, store=store, llm=FakeSummLLM(), players=["James"], snapshot=snap)
    assert ckpt == sdir / f"{sid}.checkpoint.md"
    assert ckpt.exists()
    assert not log.path.with_suffix(".summary.md").exists()
    # Active session stays open, unsummarised() still tracks it, retention won't delete it
    assert log.active_id == sid
    assert learning.unsummarised(sdir) == [log.path]
    assert apply_retention(sdir, store=store, retention_days=0) == []
    # Live chunks under <sid>.jsonl are preserved alongside <sid>.checkpoint.md chunks
    sources = {r[0] for r in store.db.execute("SELECT source FROM chunks").fetchall()}
    assert sources == {f"{sid}.jsonl", f"{sid}.checkpoint.md"}
    # Deliberate note preserved at weight 2.0
    note_weights = store.db.execute(
        "SELECT weight FROM chunks WHERE source = ? AND title LIKE '%Deliberate notes'", (f"{sid}.checkpoint.md",)
    ).fetchall()
    assert note_weights == [(2.0,)]

    # End-of-session summarise_session supersedes .checkpoint.md cleanly
    summ = summarise_session(log.path, store=store, llm=FakeSummLLM(), players=["James"])
    assert summ == sdir / f"{sid}.summary.md"
    assert not ckpt.exists()
    sources_after = {r[0] for r in store.db.execute("SELECT source FROM chunks").fetchall()}
    assert sources_after == {f"{sid}.jsonl", f"{sid}.summary.md"}


def test_checkpoint_empty_or_filler_only_returns_none(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    log = SessionLog(tmp_path / "sessions")
    log.start()
    log.append(text="Yeah.", decision="IGNORE", spoke=False)
    log.append(text="Uh.", decision="IGNORE", spoke=False)

    class FailIfCalledLLM:
        def complete(self, messages, max_tokens=1200):
            raise AssertionError("should not be called for filler-only session")

    assert checkpoint_session(log.path, store=store, llm=FailIfCalledLLM()) is None
    assert not log.path.with_suffix(".checkpoint.md").exists()


def test_campaign_include_checkpoints_folds_then_rebuilds_cleanly_on_final_summary(tmp_path):
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    (sdir / "20260916-a.summary.md").write_text("## Scenes\n- Saltmere\n", encoding="utf-8")
    raw = sdir / "20261002-b.jsonl"
    raw.write_text(json.dumps({"ts": 1, "text": "We found the Jade Halls.", "decision": "IGNORE", "spoke": False}) + "\n", encoding="utf-8")

    class MergeLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, max_tokens=1200):
            self.calls += 1
            if "SUMMARY" in messages[1]["content"]:
                return f"## Story so far\n- Merged {self.calls}\n## Open threads\n- Thread {self.calls}\n"
            return f"## Scenes\n- The Jade Halls ({self.calls})\n"

    llm = MergeLLM()
    checkpoint_session(raw, store=store, llm=llm)
    # Default campaign_due ignores checkpoints
    assert foldable(sdir, include_checkpoints=False) == [("20260916-a", foldable(sdir)[0][1])]
    assert len(foldable(sdir, include_checkpoints=True)) == 2
    merged = refresh_campaign(sdir, store=store, llm=llm, include_checkpoints=True)
    assert merged == 2
    assert [sid for sid, _ in folded(sdir / CAMPAIGN_FILE)] == ["20260916-a", "20261002-b"]

    # Final summary supersedes checkpoint and refresh_campaign rebuilds without duplicating 20261002-b
    summarise_session(raw, store=store, llm=llm)
    assert not (sdir / "20261002-b.checkpoint.md").exists()
    assert campaign_due(sdir)
    rebuilt = refresh_campaign(sdir, store=store, llm=llm)
    assert rebuilt == 2
    assert [sid for sid, _ in folded(sdir / CAMPAIGN_FILE)] == ["20260916-a", "20261002-b"]


def test_purge_session_removes_checkpoint_file_and_chunks(tmp_path):
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    raw = sdir / "20261002-x.jsonl"
    raw.write_text(json.dumps({"ts": 1, "text": "Secret vault.", "decision": "IGNORE", "spoke": False}) + "\n", encoding="utf-8")

    class LLM:
        def complete(self, messages, max_tokens=1200):
            return "## Scenes\n- Secret vault\n"

    ckpt = checkpoint_session(raw, store=store, llm=LLM())
    assert ckpt.exists()
    assert purge_session(sdir, "20261002-x", store=store) is True
    assert not ckpt.exists()
    assert store.search("vault", corpus="sessions", k=3) == []


def test_api_session_checkpoint_guards_and_full_lifecycle(tmp_path, monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(learning, "_spawn", lambda fn: fn())
    app, store = _make_app(tmp_path, fake=fake, monkeypatch=monkeypatch)
    client = app.test_client()
    try:
        # 409 when no active session
        r = client.post("/api/session/checkpoint", json={})
        assert r.status_code == 409 and r.get_json()["ok"] is False

        sid = client.post("/api/session/start").get_json()["session"]
        _turn(client, "The bell-ringer waits in the tower.")

        # 409 when unmuted & awake without force=true
        r = client.post("/api/session/checkpoint", json={})
        assert r.status_code == 409 and r.get_json()["ok"] is False

        # Mute -> checkpoint succeeds without ending session or stopping LiveIndexer
        client.post("/api/mute")
        fake.chat_replies.append("## Scenes\n- Tower\n")
        fake.chat_replies.append("## Story so far\n- Heard of the bell-ringer\n## Open threads\n- Find him\n")
        r = client.post("/api/session/checkpoint", json={})
        body = r.get_json()
        assert r.status_code == 200
        assert body["ok"] is True and body["session"] == sid
        assert body["summary_path"].endswith(f"{sid}.checkpoint.md")
        assert body["campaign_updated"] is True
        assert app.extensions["session_log"].active_id == sid
        assert app.extensions["live_index"].path is not None

        # LiveIndexer still ticks and indexes subsequent lines after unmute
        client.post("/api/unmute")
        _turn(client, "The docks smell of tar and brine.")
        app.extensions["live_index"].tick(0.0)
        app.extensions["live_index"].tick(BATCH_AGE_S)
        assert store.search("tar brine", corpus="sessions", k=1)

        # force=true works while unmuted & awake
        fake.chat_replies.append("## Scenes\n- Tower and docks\n")
        fake.chat_replies.append("## Story so far\n- Tower and docks\n")
        r_force = client.post("/api/session/checkpoint", json={"force": True})
        assert r_force.status_code == 200 and r_force.get_json()["ok"] is True

        # Ending session produces .summary.md, removes .checkpoint.md, and rebuilds campaign memory
        fake.chat_replies.append("## Scenes\n- Final tower and docks\n")
        fake.chat_replies.append("## Story so far\n- Final campaign\n")
        client.post("/api/session/end")
        assert (tmp_path / "sessions" / f"{sid}.summary.md").exists()
        assert not (tmp_path / "sessions" / f"{sid}.checkpoint.md").exists()
        assert [s for s, _ in folded(tmp_path / "sessions" / CAMPAIGN_FILE)] == [sid]
    finally:
        app.extensions["close"]()
        fake.close()


def test_api_session_checkpoint_empty_and_deep_night_and_unmute_abort(tmp_path, monkeypatch):
    fake = FakeLLM()
    app, _store = _make_app(tmp_path, fake=fake, monkeypatch=monkeypatch)
    client = app.test_client()
    try:
        sid = client.post("/api/session/start").get_json()["session"]
        # Deep Night allows checkpointing without muting
        client.post("/api/deep_night", json={"asleep": True})
        r_empty = client.post("/api/session/checkpoint", json={})
        assert r_empty.status_code == 200
        assert r_empty.get_json() == {"ok": True, "session": sid, "summary_path": None, "campaign_updated": False}

        # Add multi-window lines, then simulate unmuting while part 1 is running -> Paused before part 2 (409)
        from familiar_brain import summariser
        monkeypatch.setattr(summariser, "WINDOW_CHARS", 40)
        app.extensions["session_log"].append(
            text="A dragon flies overhead above the old watchtower.", decision="IGNORE", spoke=False
        )
        app.extensions["session_log"].append(
            text="The guards raise the alarm across the stone bridge.", decision="IGNORE", spoke=False
        )

        def unmute_mid_call(self, messages, max_tokens=1200):
            app.extensions["brain_state"].asleep = False
            return "## Scenes\n- Dragon\n"

        monkeypatch.setattr(LLMClient, "complete", unmute_mid_call)
        r_paused = client.post("/api/session/checkpoint", json={})
        assert r_paused.status_code == 409 and r_paused.get_json()["ok"] is False
        assert not (tmp_path / "sessions" / f"{sid}.checkpoint.md").exists()
    finally:
        app.extensions["close"]()
        fake.close()


# --- Deep Night (`asleep`), enriched /api/state, embedder config & thread cleanup ---


def test_deep_night_api_and_independence_from_mute(tmp_path, monkeypatch):
    fake = FakeLLM()
    app, _ = _make_app(tmp_path, fake=fake, monkeypatch=monkeypatch)
    client = app.test_client()
    try:
        # Invalid payloads -> 400
        assert client.post("/api/deep_night", json={"asleep": "yes"}).status_code == 400
        assert client.post("/api/deep_night", json={"asleep": 1}).status_code == 400
        assert client.post("/api/deep_night", json=["bad"]).status_code == 400

        # Wake first so tracking turns on
        assert ("head_tracking", {"enabled": True}) in _calls(_turn(client, "Rocky, hello"))
        assert client.get("/api/state").get_json()["tracking"] is True

        # Enter Deep Night via toggle
        r = client.post("/api/deep_night", json={})
        assert r.status_code == 200 and r.get_json() == {"asleep": True, "muted": False}

        # In Deep Night: named lines are ignored, tracking turns off + sleepy cue plays, muted remains False
        calls = _calls(_turn(client, "Rocky, what is in the chest?"))
        assert calls == [("head_tracking", {"enabled": False}), ("play_emotion", {"emotion": "sleepy"})]
        st = client.get("/api/state").get_json()
        assert st["asleep"] is True and st["muted"] is False and st["tracking"] is False

        # Muting/unmuting does not alter asleep
        client.post("/api/mute")
        assert client.get("/api/state").get_json()["asleep"] is True
        client.post("/api/unmute")
        st2 = client.get("/api/state").get_json()
        assert st2["muted"] is False and st2["asleep"] is True

        # Exit Deep Night
        r_wake = client.post("/api/deep_night", json={"asleep": False})
        assert r_wake.status_code == 200 and r_wake.get_json() == {"asleep": False, "muted": False}
        assert ("head_tracking", {"enabled": True}) in _calls(_turn(client, "Rocky, hello again"))
    finally:
        app.extensions["close"]()
        fake.close()


def test_sleep_transition_voice_commands_and_lore_safety(tmp_path, monkeypatch):
    assert sleep_transition("Deep night.") is True
    assert sleep_transition("It's deep night, everyone quiet.") is True
    assert sleep_transition("What happens during deep night?") is None
    assert sleep_transition("Tell me about deep night") is None
    assert sleep_transition("Deep night is over.") is False
    assert sleep_transition("Good morning, Rocky!") is False

    fake = FakeLLM()
    app, _ = _make_app(tmp_path, fake=fake, monkeypatch=monkeypatch)
    client = app.test_client()
    try:
        sleep_resp = _turn(client, "It's deep night.")
        assert _calls(sleep_resp) == [("play_emotion", {"emotion": "sleepy"})]
        assert client.get("/api/state").get_json()["asleep"] is True
        # Named turn stays silent while asleep
        named_asleep = _turn(client, "Rocky, wake up!")
        assert _calls(named_asleep) == [("idle_do_nothing", {})]
        assert "response.output_text.delta" not in named_asleep.get_data(as_text=True)
        # Waking phrase wakes Rocky and replies when named
        resp = _turn(client, "Good morning, Rocky!")
        assert client.get("/api/state").get_json()["asleep"] is False
        assert "response.output_text.delta" in resp.get_data(as_text=True)
    finally:
        app.extensions["close"]()
        fake.close()


def test_api_state_schema_transcript_privacy_and_hooks(tmp_path, monkeypatch):
    fake = FakeLLM()
    app, _ = _make_app(tmp_path, fake=fake, monkeypatch=monkeypatch)
    client = app.test_client()
    try:
        st0 = client.get("/api/state").get_json()
        required = {
            "muted", "asleep", "mode", "tracking", "logging", "session",
            "session_started_at", "last_spoke_at", "last_turn_ms",
            "recent", "transcript", "speakers", "robot",
        }
        assert required <= set(st0.keys())
        assert st0["session_started_at"] is None and st0["last_turn_ms"] is None
        assert st0["speakers"] == {
            "enabled": False, "enrolled": [], "last_speaker": None,
            "last_confidence": None, "capturing": False,
        }
        assert st0["robot"] == {
            "backend_connected": None, "host": None, "last_checked_at": None,
            "detail": "not checked", "disabled": False,
            "mode": "legacy", "mic_level": 0.0,
        }

        client.post("/api/session/start")
        _turn(client, "Rocky, hello!")
        st1 = client.get("/api/state").get_json()
        assert st1["session_started_at"] is not None
        assert isinstance(st1["last_turn_ms"], float)
        assert len(st1["transcript"]) == 1
        assert set(st1["transcript"][0].keys()) == {"text", "speaker", "decision", "spoke", "ts"}

        # Private speech while logging=False must stay out of transcript too
        client.post("/api/mute_and_stop_logging")
        _turn(client, "Private secret table note")
        assert len(client.get("/api/state").get_json()["transcript"]) == 1

        # Extension hooks for M3/M4
        app.extensions["speaker_status"] = lambda: {
            "enabled": True, "enrolled": ["DM", "Brokk"], "last_speaker": "Brokk",
            "last_confidence": 0.91, "capturing": False,
        }
        app.extensions["robot_status"] = lambda: {
            "backend_connected": True, "host": "reachy-mini.local",
            "last_checked_at": 123.0, "detail": "ok", "disabled": False,
        }
        st2 = client.get("/api/state").get_json()
        assert st2["speakers"]["enrolled"] == ["DM", "Brokk"]
        assert st2["robot"]["backend_connected"] is True
    finally:
        app.extensions["close"]()
        fake.close()


def test_public_deterministic_embedder_config_and_zero_thread_leak(tmp_path, monkeypatch):
    emb = DeterministicEmbedder()
    v1, v2, v3 = emb.embed(["grapple the goblin", "grapple a goblin", "cast fireball"])
    assert len(v1) == 384 and emb.dim == 384
    assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-5)
    assert cosine(v1, v2) > cosine(v1, v3)
    # Empty / punctuation-only text still yields normalized 384-d unit vector
    v_empty = emb.embed(["..."])[0]
    assert len(v_empty) == 384 and math.isclose(sum(x * x for x in v_empty), 1.0, rel_tol=1e-5)

    # FAMILIAR_EMBEDDER env override works without monkeypatching indexing.open_store
    monkeypatch.setenv("FAMILIAR_EMBEDDER", "deterministic")
    fake = FakeLLM()
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "notes" / "brokk.md").write_text("# Brokk\nBrokk Trenevain is an artificer.\n", encoding="utf-8")
    app, _ = _make_app(tmp_path, fake=fake, use_deterministic=False)
    client = app.test_client()
    try:
        assert isinstance(app.extensions["store"].embedder, DeterministicEmbedder)
        client.post("/api/session/start")
        assert any(t.name == "live-index" for t in threading.enumerate())
        _turn(client, "Rocky, who is Brokk?")
        sent_instr = fake.requests[-1]["body"]["instructions"]
        assert "artificer" in sent_instr
    finally:
        app.extensions["close"]()
        fake.close()

    assert [t for t in threading.enumerate() if t.name == "live-index"] == []


def test_checkpoint_session_race_against_summarise_session_raises_paused_with_no_orphan(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    sdir = tmp_path / "sessions"
    log = SessionLog(sdir)
    sid = log.start()
    log.append(text="The bell-ringer hides a silver key.", decision="IGNORE", spoke=False)
    raw_path = log.path

    class RacingLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, max_tokens=1200):
            self.calls += 1
            if self.calls == 1:
                # Simulate final summarise_session landing before checkpoint acquires _LOCK
                summarise_session(raw_path, store=store, llm=self)
                return "## Scenes\n- Orphan checkpoint\n"
            return "## Scenes\n- Final summary\n"

    with pytest.raises(Paused):
        checkpoint_session(raw_path, store=store, llm=RacingLLM(), snapshot=log.snapshot_text())
    assert not raw_path.with_suffix(".checkpoint.md").exists()
    assert raw_path.with_suffix(".summary.md").exists()
    assert store.db.execute("SELECT COUNT(*) FROM chunks WHERE source = ?", (f"{sid}.checkpoint.md",)).fetchone()[0] == 0

    # Also verify Purged is raised if raw_path is deleted mid-checkpoint
    log2 = SessionLog(sdir)
    sid2 = log2.start()
    log2.append(text="A hidden vault opens.", decision="IGNORE", spoke=False)
    raw2 = log2.path

    class PurgeRacingLLM:
        def complete(self, messages, max_tokens=1200):
            purge_session(sdir, sid2, store=store)
            return "## Scenes\n- Vault\n"

    with pytest.raises(Purged):
        checkpoint_session(raw2, store=store, llm=PurgeRacingLLM(), snapshot=log2.snapshot_text())
    assert not raw2.with_suffix(".checkpoint.md").exists()


def test_api_checkpoint_force_validation_and_concurrent_session_end_race(tmp_path, monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(learning, "_spawn", lambda fn: fn())
    app, store = _make_app(tmp_path, fake=fake, monkeypatch=monkeypatch)
    client = app.test_client()
    try:
        sid = client.post("/api/session/start").get_json()["session"]
        _turn(client, "The tavern catches fire.")
        client.post("/api/mute")

        # Non-bool force or non-object body -> 400
        assert client.post("/api/session/checkpoint", json={"force": "false"}).status_code == 400
        assert client.post("/api/session/checkpoint", json={"force": 1}).status_code == 400
        assert client.post("/api/session/checkpoint", json=["bad"]).status_code == 400

        entered = threading.Event()
        release = threading.Event()

        def slow_complete(self, messages, max_tokens=1200):
            if not entered.is_set():
                entered.set()
                release.wait(timeout=5.0)
                return "## Scenes\n- Checkpoint scenes\n"
            return "## Scenes\n- Final scenes\n"

        monkeypatch.setattr(LLMClient, "complete", slow_complete)
        res_holder: dict = {}

        def run_ckpt():
            c2 = app.test_client()
            r = c2.post("/api/session/checkpoint", json={"force": False})
            res_holder["status"] = r.status_code
            res_holder["body"] = r.get_json()

        t = threading.Thread(target=run_ckpt)
        t.start()
        assert entered.wait(timeout=5.0)

        # End session while checkpoint is blocked in LLMClient.complete()
        r_end = client.post("/api/session/end")
        assert r_end.status_code == 200
        release.set()
        t.join(timeout=5.0)

        assert res_holder["status"] == 409
        assert res_holder["body"]["ok"] is False
        assert not (tmp_path / "sessions" / f"{sid}.checkpoint.md").exists()
        assert (tmp_path / "sessions" / f"{sid}.summary.md").exists()
        assert store.db.execute(
            "SELECT COUNT(*) FROM chunks WHERE source = ?", (f"{sid}.checkpoint.md",)
        ).fetchone()[0] == 0
    finally:
        app.extensions["close"]()
        fake.close()


def test_speak_now_rejected_while_muted_or_asleep_and_never_leaks(tmp_path, monkeypatch):
    fake = FakeLLM()
    app, _ = _make_app(tmp_path, fake=fake, monkeypatch=monkeypatch)
    client = app.test_client()
    try:
        # While muted -> 409 and does not arm
        client.post("/api/mute")
        r_muted = client.post("/api/speak_now")
        assert r_muted.status_code == 409 and r_muted.get_json()["ok"] is False
        client.post("/api/unmute")
        assert "response.output_text.delta" not in _turn(client, "Pass me the d6 please.").get_data(as_text=True)

        # While in Deep Night -> 409 and does not arm
        client.post("/api/deep_night", json={"asleep": True})
        r_asleep = client.post("/api/speak_now")
        assert r_asleep.status_code == 409 and r_asleep.get_json()["ok"] is False
        client.post("/api/deep_night", json={"asleep": False})
        assert "response.output_text.delta" not in _turn(client, "Pass me the d8 please.").get_data(as_text=True)

        # Armed while awake, then deep_night(True) -> deep_night(False) -> disarmed
        assert client.post("/api/speak_now").status_code == 200
        client.post("/api/deep_night", json={"asleep": True})
        client.post("/api/deep_night", json={"asleep": False})
        assert "response.output_text.delta" not in _turn(client, "Nobody named Rocky here.").get_data(as_text=True)
    finally:
        app.extensions["close"]()
        fake.close()


def test_apply_retention_zero_negative_and_keep_marker_protection(tmp_path):
    import os
    import time

    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    raw1 = sdir / "20260916-a.jsonl"
    raw2 = sdir / "20260916-b.jsonl"
    for raw in (raw1, raw2):
        raw.write_text(json.dumps({"ts": 1, "text": "Silver key.", "decision": "IGNORE", "spoke": False}) + "\n", encoding="utf-8")
        raw.with_suffix(".summary.md").write_text("## Scenes\n- Silver key\n", encoding="utf-8")
        old = time.time() - 30 * 86400
        os.utime(raw, (old, old))
    (sdir / "20260916-a.keep").write_text("protected\n", encoding="utf-8")

    try:
        assert apply_retention(sdir, store=store, retention_days=0) == []
        assert apply_retention(sdir, store=store, retention_days=-5) == []
        assert raw1.exists() and raw2.exists()

        removed = apply_retention(sdir, store=store, retention_days=7)
        assert removed == ["20260916-b.jsonl"]
        assert raw1.exists() and not raw2.exists()
    finally:
        store.close()


def test_concurrent_checkpoint_end_and_immediate_delete_stress(tmp_path, monkeypatch):
    fake = FakeLLM()
    reply_lock = threading.Lock()
    counter = {"n": 0}

    def thread_safe_complete(self, messages, max_tokens=1200):
        with reply_lock:
            counter["n"] += 1
            n = counter["n"]
        sys_msg = messages[0]["content"].lower()
        if "campaign memory" in sys_msg:
            return (
                f"## Story so far\n- Vault {n}\n## NPCs\n- none\n## Places\n- Vault\n"
                f"## Open threads\n- Thread {n}\n## Loot and resources\n- none\n"
                "## Decisions that matter\n- none\n"
            )
        return f"## Scenes\n- Vault {n}\n## Open threads\n- Thread {n}\n"

    monkeypatch.setattr(LLMClient, "complete", thread_safe_complete)
    app, _store = _make_app(tmp_path, fake=fake, monkeypatch=monkeypatch)
    app.config["TESTING"] = False
    try:
        for i in range(20):
            c_start = app.test_client()
            r_start = c_start.post("/api/session/start")
            assert r_start.status_code == 200
            sid = r_start.get_json()["session"]
            app.extensions["session_log"].append(
                text=f"We open the silver vault door in trial {i}.",
                decision="IGNORE",
                spoke=False,
            )

            statuses: dict[str, int] = {}
            errors: list[BaseException] = []

            if i % 2 == 0:
                barrier = threading.Barrier(2)

                def do_ckpt():
                    try:
                        barrier.wait(timeout=5.0)
                        r = app.test_client().post("/api/session/checkpoint", json={"force": True})
                        statuses["checkpoint"] = r.status_code
                    except BaseException as exc:
                        errors.append(exc)

                def do_end():
                    try:
                        barrier.wait(timeout=5.0)
                        r = app.test_client().post("/api/session/end")
                        statuses["end"] = r.status_code
                    except BaseException as exc:
                        errors.append(exc)

                t1 = threading.Thread(target=do_ckpt)
                t2 = threading.Thread(target=do_end)
                t1.start()
                t2.start()
                t1.join(timeout=5.0)
                t2.join(timeout=5.0)
                # Fire DELETE immediately while real Learner background thread is draining
                r_del = app.test_client().delete(f"/api/session/{sid}")
                statuses["delete"] = r_del.status_code
            else:
                r_ckpt = app.test_client().post("/api/session/checkpoint", json={"force": True})
                statuses["checkpoint"] = r_ckpt.status_code
                barrier = threading.Barrier(2)

                def do_end_only():
                    try:
                        barrier.wait(timeout=5.0)
                        r = app.test_client().post("/api/session/end")
                        statuses["end"] = r.status_code
                    except BaseException as exc:
                        errors.append(exc)

                def do_del_race():
                    try:
                        barrier.wait(timeout=5.0)
                        r = app.test_client().delete(f"/api/session/{sid}")
                        statuses["delete"] = r.status_code
                    except BaseException as exc:
                        errors.append(exc)

                t1 = threading.Thread(target=do_end_only)
                t2 = threading.Thread(target=do_del_race)
                t1.start()
                t2.start()
                t1.join(timeout=5.0)
                t2.join(timeout=5.0)
                if statuses.get("delete") == 409:
                    r_del2 = app.test_client().delete(f"/api/session/{sid}")
                    statuses["delete_retry"] = r_del2.status_code

            assert errors == []
            assert statuses["checkpoint"] in (200, 409)
            assert statuses["end"] == 200
            assert statuses["delete"] in (200, 409)
            if "delete_retry" in statuses:
                assert statuses["delete_retry"] == 200
            assert 500 not in statuses.values()
    finally:
        app.extensions["close"]()
        fake.close()

