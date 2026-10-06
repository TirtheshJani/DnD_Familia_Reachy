"""Catch-up learning: a session that was never ended is still summarised, one at a time,
never while a session is going, and its raw log is never lost before it has been."""
import io
import json
import logging
import os
import time
from pathlib import Path

import requests

from familiar_brain import cli, indexing
from familiar_brain.learning import Learner, learn_all, unsummarised
from familiar_brain.retrieval import Store
from familiar_brain.summariser import apply_retention
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm_server import FakeLLM as FakeLLMServer

SUMMARY = "## Scenes\n- The cistern\n## NPCs met\n- The barkeep\n"


class FakeLLM:
    def __init__(self, on_call=None, fail=None):
        self.calls = 0
        self.on_call = on_call
        self.fail = fail

    def complete(self, messages, max_tokens=1200):
        self.calls += 1
        if self.on_call:
            self.on_call()
        if self.fail:
            raise self.fail
        return SUMMARY


def _raw(d: Path, sid: str, text: str = "The cistern is dark.", age_days: float = 0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sid}.jsonl"
    p.write_text(json.dumps({"ts": 1, "text": text, "decision": "IGNORE", "spoke": False}) + "\n",
                 encoding="utf-8")
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(p, (old, old))
    return p


def _learner(tmp, llm, active=None, players=None):
    store = Store(tmp / "k.db", embedder=FakeEmbedder())
    state = {"active": active}
    lr = Learner(tmp / "sessions", store=store, llm=llm,
                 players_path=players or tmp / "players.txt",
                 active=lambda: state["active"], spawn=lambda fn: fn())
    return lr, store, state


def test_catch_up_picks_every_log_with_no_summary_except_the_active_one(tmp_path):
    d = tmp_path / "sessions"
    _raw(d, "20260911-a"); _raw(d, "20260916-b"); _raw(d, "20261002-live")
    _raw(d, "20260910-done"); (d / "20260910-done.summary.md").write_text(SUMMARY)
    assert [p.stem for p in unsummarised(d, active="20261002-live")] == ["20260911-a", "20260916-b"]


def test_catch_up_summarises_each_waiting_session_in_turn(tmp_path):
    _raw(tmp_path / "sessions", "s1"); _raw(tmp_path / "sessions", "s2")
    lr, store, _ = _learner(tmp_path, FakeLLM())
    assert lr.catch_up() == ["s1", "s2"]
    assert (tmp_path / "sessions" / "s1.summary.md").exists()
    assert (tmp_path / "sessions" / "s2.summary.md").exists()
    assert store.search("barkeep", corpus="sessions", k=1)


def test_nothing_is_summarised_while_a_session_is_active(tmp_path):
    _raw(tmp_path / "sessions", "s1")
    llm = FakeLLM()
    lr, _, state = _learner(tmp_path, llm, active="s9")
    assert lr.catch_up() == ["s1"]
    assert llm.calls == 0 and lr.waiting() == ["s1"]
    state["active"] = None
    lr.kick()
    # The summary, then its merge into the campaign memory (deviation P18).
    assert llm.calls == 2 and lr.waiting() == []


def test_a_session_starting_mid_catch_up_finishes_the_current_one_and_pauses_the_rest(tmp_path):
    for s in ("s1", "s2", "s3"):
        _raw(tmp_path / "sessions", s)
    state_ref = {}
    llm = FakeLLM(on_call=lambda: state_ref["state"].update(active="s9"))
    lr, _, state = _learner(tmp_path, llm)
    state_ref["state"] = state
    lr.catch_up()
    assert llm.calls == 1 and (tmp_path / "sessions" / "s1.summary.md").exists()
    assert lr.waiting() == ["s2", "s3"]


def test_a_failed_summary_keeps_the_raw_log_for_the_next_catch_up(tmp_path):
    raw = _raw(tmp_path / "sessions", "s1")
    lr, _, _ = _learner(tmp_path, FakeLLM(fail=requests.ConnectionError("refused")))
    lr.catch_up()
    assert raw.exists() and not raw.with_suffix(".summary.md").exists()
    assert unsummarised(tmp_path / "sessions") == [raw]


def test_a_purged_session_still_queued_is_skipped(tmp_path):
    raw = _raw(tmp_path / "sessions", "s1")
    llm = FakeLLM()
    lr, _, state = _learner(tmp_path, llm, active="s9")
    lr.catch_up()
    raw.unlink()
    state["active"] = None
    lr.kick()
    assert llm.calls == 0 and lr.waiting() == []


def test_a_missing_players_file_is_warned_about_before_every_summary(tmp_path, caplog):
    _raw(tmp_path / "sessions", "s1"); _raw(tmp_path / "sessions", "s2")
    lr, _, _ = _learner(tmp_path, FakeLLM())
    with caplog.at_level(logging.WARNING, logger="familiar.learning"):
        lr.catch_up()
    warned = [r for r in caplog.records if "will not be scrubbed" in r.message]
    assert len(warned) == 2 and all(r.levelno == logging.WARNING for r in warned)


def test_a_players_file_is_read_for_each_summary(tmp_path, caplog):
    _raw(tmp_path / "sessions", "s1")
    players = tmp_path / "players.txt"
    players.write_text("Barkeep\n", encoding="utf-8")
    lr, _, _ = _learner(tmp_path, FakeLLM(), players=players)
    with caplog.at_level(logging.WARNING, logger="familiar.learning"):
        lr.catch_up()
    assert not any("will not be scrubbed" in r.message for r in caplog.records)
    assert "barkeep" not in (tmp_path / "sessions" / "s1.summary.md").read_text().lower()


def test_retention_never_deletes_a_raw_log_that_has_no_summary(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    raw = _raw(tmp_path / "sessions", "s1", age_days=30)
    assert apply_retention(raw.parent, store=store, retention_days=7) == []
    assert raw.exists()


def test_retention_leaves_a_log_the_learner_still_holds(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    raw = _raw(tmp_path / "sessions", "s1", age_days=30)
    raw.with_suffix(".summary.md").write_text(SUMMARY)
    assert apply_retention(raw.parent, store=store, retention_days=7, keep={raw.name}) == []
    assert apply_retention(raw.parent, store=store, retention_days=7) == [raw.name]


def test_learn_all_reports_progress_and_summarises_everything(tmp_path):
    _raw(tmp_path / "sessions", "s1"); _raw(tmp_path / "sessions", "s2")
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    out = io.StringIO()
    failed = learn_all(tmp_path / "sessions", store=store, llm=FakeLLM(),
                       players_path=tmp_path / "players.txt", out=out)
    text = out.getvalue()
    assert failed == 0 and "1/2 s1" in text and "2/2 s2" in text and "learnt 2 of 2" in text
    assert (tmp_path / "sessions" / "s2.summary.md").exists()


def test_learn_all_with_the_model_down_says_so_and_keeps_the_raw_log(tmp_path):
    raw = _raw(tmp_path / "sessions", "s1")
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    out = io.StringIO()
    failed = learn_all(tmp_path / "sessions", store=store,
                       llm=FakeLLM(fail=requests.ConnectionError("Connection refused")),
                       players_path=tmp_path / "players.txt", out=out)
    assert failed == 1 and raw.exists() and not raw.with_suffix(".summary.md").exists()
    assert "could not reach the model" in out.getvalue() and "raw log is kept" in out.getvalue()


def test_learn_all_with_nothing_to_learn_says_so(tmp_path):
    (tmp_path / "sessions").mkdir()
    out = io.StringIO()
    assert learn_all(tmp_path / "sessions", store=Store(tmp_path / "k.db", embedder=FakeEmbedder()),
                     llm=FakeLLM(), players_path=tmp_path / "p.txt", out=out) == 0
    assert "nothing to learn" in out.getvalue()


def test_cli_learn_runs_against_the_configured_model(tmp_path, monkeypatch, capsys):
    fake = FakeLLMServer()
    try:
        fake.chat_replies.append(SUMMARY)
        _raw(tmp_path / "sessions", "s1")
        store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
        monkeypatch.setattr(indexing, "open_store", lambda cfg: store)
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
persona = "unused"
sessions_dir = "{(tmp_path / 'sessions').as_posix()}"
players = "{(tmp_path / 'players.txt').as_posix()}"
[knowledge]
db_path = "{(tmp_path / 'k.db').as_posix()}"
notes_dir = "{(tmp_path / 'notes').as_posix()}"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
""")
        cli.main(["learn", "--config", str(cfg)])
    finally:
        fake.close()
    assert "learnt 1 of 1" in capsys.readouterr().out
    assert (tmp_path / "sessions" / "s1.summary.md").read_text() == SUMMARY
    assert fake.requests[-1]["body"]["id_slot"] == 0


def _long_raw(d: Path, sid: str, chars: int) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sid}.jsonl"
    rows, total, i = [], 0, 0
    while total < chars:
        # Each line its own words, so no line reads as a revision of the one before.
        text = " ".join(f"w{i}x{j}" for j in range(30)) + "."
        rows.append(json.dumps({"ts": i, "text": text, "decision": "IGNORE", "spoke": False}))
        total += len(text) + 1; i += 1
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_a_purge_landing_mid_summary_leaves_nothing_behind(tmp_path):
    """The summary's model calls take minutes. A DELETE /api/session/<id> landing during them
    deleted the log and its chunks, then the summary wrote its file and re-added the summary
    and transcript chunks, and the campaign merge folded the purged session back in."""
    from familiar_brain.summariser import purge_session
    from familiar_brain.transcript import dedupe_revisions, piece_chunks, read_rows, transcript_pieces
    d = tmp_path / "sessions"
    raw = _raw(d, "20261002-aaa", text="Bram hides the key.")
    lr_ref = {}
    llm = FakeLLM(on_call=lambda: purge_session(d, "20261002-aaa", store=lr_ref["store"])
                  if llm.calls == 1 else None)
    lr, store, _ = _learner(tmp_path, llm)
    lr_ref["store"] = store
    store.add_chunks(piece_chunks(transcript_pieces(dedupe_revisions(read_rows(raw))), sid=raw.stem,
                                  source=raw.name))
    lr.catch_up()
    assert list(d.iterdir()) == []  # no summary, no campaign memory
    assert store.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert lr.waiting() == []


def test_no_call_goes_out_once_a_session_starts_mid_summary(tmp_path):
    """Only the campaign merge looked at the active session. A long session summarised
    window by window sent every partial and the merge after a session had started, each up
    to 1200 tokens, on the GPU the table's replies wait for."""
    from familiar_brain.summariser import WINDOW_CHARS
    raw = _long_raw(tmp_path / "sessions", "s1", WINDOW_CHARS * 4)
    state_ref = {}
    # The session starts during the first call only, so the second run can finish.
    llm = FakeLLM(on_call=lambda: state_ref["state"].update(active="s9") if llm.calls == 1 else None)
    lr, store, state = _learner(tmp_path, llm)
    state_ref["state"] = state
    lr.catch_up()
    assert llm.calls == 1
    assert not raw.with_suffix(".summary.md").exists() and lr.waiting() == ["s1"]
    assert store.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0  # nothing written
    state["active"] = None
    lr.kick()
    assert raw.with_suffix(".summary.md").exists() and lr.waiting() == []


def test_a_session_learnt_while_catch_up_listed_it_is_not_learnt_again(tmp_path, monkeypatch):
    """catch_up lists the unsummarised logs outside the lock. If the worker finished one of
    them in between, it was queued again and summarised a second time."""
    from familiar_brain import learning
    raw = _raw(tmp_path / "sessions", "s1")
    llm = FakeLLM()
    lr, _, _ = _learner(tmp_path, llm)
    real = learning.unsummarised

    def racy(d, active=None):
        pending = real(d, active=active)
        lr.queue(pending); lr.kick()  # the worker finishes them before the listing is used
        return pending

    monkeypatch.setattr(learning, "unsummarised", racy)
    lr.catch_up()
    assert raw.with_suffix(".summary.md").exists()
    assert llm.calls == 2  # the summary and its campaign merge, once each
