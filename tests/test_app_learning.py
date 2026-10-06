"""Rocky keeps learning without anyone calling /api/session/end: the session in progress is
indexed as it goes, and every session left unsummarised is learnt at the next idle moment.

The learner runs synchronously here (learning._spawn) and the live indexer's worker is
never started (live_index._spawn); its ticks are driven by hand with a clock reading.
"""
import json
import logging
from pathlib import Path

import pytest

from familiar_brain import indexing, learning, live_index
from familiar_brain.app import create_app
from familiar_brain.live_index import BATCH_AGE_S
from familiar_brain.retrieval import Store
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm_server import FakeLLM

SUMMARY = "## Scenes\n- The cistern\n## NPCs met\n- The barkeep\n"


@pytest.fixture
def make(tmp_path: Path, monkeypatch):
    fake = FakeLLM()
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    monkeypatch.setattr(indexing, "open_store", lambda cfg: store)
    monkeypatch.setattr(learning, "_spawn", lambda fn: fn())
    monkeypatch.setattr(live_index, "_spawn", lambda fn: None)
    (tmp_path / "persona.yaml").write_text("name: Understudy\naliases: []\ntts_speaker: A\nmanner: dry\n")
    (tmp_path / "notes").mkdir()
    (tmp_path / "sessions").mkdir()
    (tmp_path / "brain.toml").write_text(f"""
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
persona = "{(tmp_path / 'persona.yaml').as_posix()}"
sessions_dir = "{(tmp_path / 'sessions').as_posix()}"
players = "{(tmp_path / 'players.txt').as_posix()}"
[knowledge]
db_path = "{(tmp_path / 'k.db').as_posix()}"
notes_dir = "{(tmp_path / 'notes').as_posix()}"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
""")

    def _make():
        app = create_app(tmp_path / "brain.toml")
        app.config["TESTING"] = True
        return app

    yield _make, fake, store, tmp_path
    fake.close()


def _orphan(tmp: Path, sid: str, text: str = "The barkeep hides a key under the cistern.") -> Path:
    p = tmp / "sessions" / f"{sid}.jsonl"
    p.write_text(json.dumps({"ts": 1, "text": text, "decision": "IGNORE", "spoke": False}) + "\n",
                 encoding="utf-8")
    return p


def _table(client, text):
    client.post("/v1/responses", json={"input": [{"role": "user", "content": text}],
                                       "stream": True}).get_data()


def _chat_calls(fake):
    return [r for r in fake.requests if r["path"].endswith("/chat/completions") and r["body"]["id_slot"] == 0]


def test_startup_learns_a_session_that_was_never_ended(make):
    build, fake, store, tmp = make
    raw = _orphan(tmp, "20260916-182232-0189cc")
    fake.chat_replies.append(SUMMARY)
    build()
    assert raw.with_suffix(".summary.md").read_text() == SUMMARY
    assert store.search("barkeep", corpus="sessions", k=1)


def test_startup_warns_when_there_is_no_players_file(make, caplog):
    build, _, _, _ = make
    with caplog.at_level(logging.WARNING, logger="familiar.learning"):
        build()
    assert any("will not be scrubbed" in r.message for r in caplog.records)


def test_startup_keeps_an_old_raw_log_its_summary_failed_for(make):
    import os, time
    build, fake, _, tmp = make
    raw = _orphan(tmp, "20260901-000000-aaaaaa")
    old = time.time() - 30 * 86400
    os.utime(raw, (old, old))
    fake.close(); fake.server.server_close()  # llama-server down: the summary fails
    build()
    assert raw.exists() and not raw.with_suffix(".summary.md").exists()


def test_starting_a_session_while_one_is_active_ends_it_and_queues_it(make):
    build, fake, _, tmp = make
    client = build().test_client()
    first = client.post("/api/session/start").get_json()["session"]
    _table(client, "The barkeep hides a key under the cistern.")
    body = client.post("/api/session/start").get_json()
    assert body["ended"] == first and body["session"] != first
    # Queued, not run: the new session is active, so the GPU stays the table's.
    assert _chat_calls(fake) == []
    assert client.post("/api/learn").get_json() == {"queued": [first], "paused": True}
    fake.chat_replies.append(SUMMARY)
    client.post("/api/session/end")
    assert (tmp / "sessions" / f"{first}.summary.md").exists()


def test_api_learn_queues_the_catch_up_and_reports_it(make):
    build, fake, _, tmp = make
    app = build()
    raw = _orphan(tmp, "20260916-182232-0189cc")  # appeared after startup
    fake.chat_replies.append(SUMMARY)
    body = app.test_client().post("/api/learn").get_json()
    assert body == {"queued": ["20260916-182232-0189cc"], "paused": False}
    assert raw.with_suffix(".summary.md").exists()


def test_the_session_so_far_is_searchable_mid_session(make):
    build, fake, store, _ = make
    app = build()
    client = app.test_client()
    client.post("/api/session/start")
    _table(client, "The innkeeper says the well behind the mill is poisoned.")
    live = app.extensions["live_index"]
    live.tick(0.0); live.tick(BATCH_AGE_S)
    hits = store.search("innkeeper well poisoned", corpus="sessions", k=1)
    assert hits and "poisoned" in hits[0].text
    assert _chat_calls(fake) == []  # embedding only


def test_a_note_mid_session_is_searchable_on_the_next_tick(make):
    build, _, store, _ = make
    app = build()
    client = app.test_client()
    client.post("/api/session/start")
    client.post("/api/note", json={"text": "Bram owes us ten gold."})
    app.extensions["live_index"].tick(0.0)
    hit = store.search("Bram gold", corpus="sessions", k=1)[0]
    assert hit.meta.get("deliberate") is True


def test_lines_said_while_logging_is_off_are_never_indexed(make):
    build, _, store, _ = make
    app = build()
    client = app.test_client()
    client.post("/api/session/start")
    client.post("/api/mute_and_stop_logging")
    _table(client, "This is between us, the secret door is behind the tapestry.")
    live = app.extensions["live_index"]
    live.tick(0.0); live.tick(BATCH_AGE_S)
    assert store.search("secret door tapestry", corpus="sessions", k=3) == []


def test_ending_a_session_replaces_its_live_chunks_with_the_summary(make):
    build, fake, store, tmp = make
    app = build()
    client = app.test_client()
    sid = client.post("/api/session/start").get_json()["session"]
    _table(client, "The barkeep hides a key under the cistern.")
    app.extensions["live_index"].tick(0.0)
    fake.chat_replies.append(SUMMARY)
    client.post("/api/session/end")
    rows = store.db.execute("SELECT source, weight FROM chunks WHERE corpus = 'sessions' ORDER BY id").fetchall()
    assert (f"{sid}.jsonl", 1.0) in rows and rows.count((f"{sid}.jsonl", 1.0)) == 1
    assert any(s == f"{sid}.summary.md" for s, _ in rows)


def test_purging_the_session_in_progress_is_refused(make):
    build, _, _, tmp = make
    client = build().test_client()
    sid = client.post("/api/session/start").get_json()["session"]
    resp = client.delete(f"/api/session/{sid}")
    assert resp.status_code == 409
    assert (tmp / "sessions" / f"{sid}.jsonl").exists()


def test_purging_a_session_rebuilds_the_campaign_memory_from_the_rest(make):
    from familiar_brain.campaign import CAMPAIGN_FILE, CAMPAIGN_PROMPT, folded
    build, fake, _, tmp = make
    _orphan(tmp, "20260916-000000-aaaaaa", "The well is poisoned.")
    _orphan(tmp, "20260923-000000-bbbbbb", "The crypt is cold.")
    fake.chat_replies.extend([SUMMARY, "## Story so far\n- A\n", SUMMARY, "## Story so far\n- A, B\n"])
    client = build().test_client()
    mem = tmp / "sessions" / CAMPAIGN_FILE
    assert [s for s, _ in folded(mem)] == ["20260916-000000-aaaaaa", "20260923-000000-bbbbbb"]
    # The campaign memory and its .prev are not sessions.
    assert [s["id"] for s in client.get("/api/sessions").get_json()["sessions"]] == [
        "20260916-000000-aaaaaa", "20260923-000000-bbbbbb"]
    fake.chat_replies.append("## Story so far\n- B\n")
    client.delete("/api/session/20260916-000000-aaaaaa")
    assert [s for s, _ in folded(mem)] == ["20260923-000000-bbbbbb"]
    last = _chat_calls(fake)[-1]["body"]["messages"]
    assert last[0]["content"] == CAMPAIGN_PROMPT and "nothing yet" in last[1]["content"]
