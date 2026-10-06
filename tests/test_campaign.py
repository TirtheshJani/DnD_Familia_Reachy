"""The running campaign memory (deviation P18): one file in data/sessions, merged by the
model from the previous memory and each new session summary, indexed above every single
session, and rebuilt from the summaries that remain when one is purged or re-summarised."""
import json
import logging
from pathlib import Path

import pytest

from familiar_brain import campaign
from familiar_brain.campaign import (
    CAMPAIGN_CAP,
    CAMPAIGN_FILE,
    CAMPAIGN_PROMPT,
    CAMPAIGN_WEIGHT,
    PREV_FILE,
    campaign_due,
    refresh_campaign,
)
from familiar_brain.knowledge import build_context
from familiar_brain.learning import Learner, learn_all
from familiar_brain.retrieval import Chunk, Store
from familiar_brain.summariser import EMPTY_SUMMARY, purge_session
from tests.fakes.embedder import FakeEmbedder

S1 = "## Scenes\n- The party reached Saltmere\n## Open threads\n- Who poisoned the well?\n"
S2 = "## Scenes\n- The crypt under the chapel\n## Open threads\n- Find the bell-ringer\n"


def memory(tag: str) -> str:
    return (f"## Story so far\n- Story {tag}\n## NPCs\n- Mother Ysolde\n## Places\n- Saltmere\n"
            f"## Open threads\n- Thread {tag}\n## Loot and resources\n- none\n## Decisions that matter\n- none\n")


class MergeLLM:
    """Records the campaign merge calls; each reply is a memory tagged with the call number."""

    def __init__(self, reply=None, fail=None):
        self.calls = []
        self.reply = reply
        self.fail = fail

    def complete(self, messages, max_tokens=1200):
        self.calls.append(messages)
        if self.fail:
            raise self.fail
        return self.reply(messages) if self.reply else memory(str(len(self.calls)))

    def user(self, i=-1) -> str:
        return self.calls[i][1]["content"]


def _sessions(tmp: Path, **summaries: str) -> Path:
    d = tmp / "sessions"; d.mkdir(exist_ok=True)
    for sid, text in summaries.items():
        (d / f"{sid}.summary.md").write_text(text, encoding="utf-8")
    return d


def _store(tmp: Path) -> Store:
    return Store(tmp / "k.db", embedder=FakeEmbedder())


def _rows(store: Store, source: str = CAMPAIGN_FILE):
    return store.db.execute("SELECT title, text, weight FROM chunks WHERE source = ?", (source,)).fetchall()


def test_the_first_summary_starts_the_campaign_memory(tmp_path):
    d = _sessions(tmp_path, **{"20260916-a": S1})
    llm = MergeLLM()
    assert refresh_campaign(d, store=_store(tmp_path), llm=llm) == 1
    assert llm.calls[0][0]["content"] == CAMPAIGN_PROMPT
    assert S1 in llm.user() and "nothing yet" in llm.user()
    assert memory("1") in (d / CAMPAIGN_FILE).read_text()
    assert not (d / PREV_FILE).exists()


def test_the_next_summary_is_merged_with_the_previous_memory_and_the_old_one_kept(tmp_path):
    d = _sessions(tmp_path, **{"20260916-a": S1})
    store, llm = _store(tmp_path), MergeLLM()
    refresh_campaign(d, store=store, llm=llm)
    first = (d / CAMPAIGN_FILE).read_text()
    _sessions(tmp_path, **{"20260923-b": S2})
    assert refresh_campaign(d, store=store, llm=llm) == 1
    assert memory("1") in llm.user() and S2 in llm.user() and S1 not in llm.user()
    assert (d / PREV_FILE).read_text() == first
    assert memory("2") in (d / CAMPAIGN_FILE).read_text()


def test_nothing_new_means_no_call(tmp_path):
    d = _sessions(tmp_path, **{"20260916-a": S1, "20260917-quiet": EMPTY_SUMMARY})
    store, llm = _store(tmp_path), MergeLLM()
    refresh_campaign(d, store=store, llm=llm)
    assert len(llm.calls) == 1  # the session where nothing was said is not folded
    assert not campaign_due(d)
    assert refresh_campaign(d, store=store, llm=llm) == 0 and len(llm.calls) == 1


def test_the_prompt_is_fiction_only_capped_and_defers_to_the_dms_notes():
    p = CAMPAIGN_PROMPT.lower()
    assert "player" in p and "out of character" in p and f"{CAMPAIGN_CAP:,}" in CAMPAIGN_PROMPT
    assert "notes" in p and "unresolved" in p
    for h in ("Story so far", "NPCs", "Places", "Open threads", "Loot and resources", "Decisions that matter"):
        assert f"## {h}" in CAMPAIGN_PROMPT


def test_a_memory_over_the_cap_is_cut_with_a_warning(tmp_path, caplog):
    d = _sessions(tmp_path, **{"20260916-a": S1})
    llm = MergeLLM(reply=lambda m: "## Story so far\n" + "- The road went on and on.\n" * 600)
    with caplog.at_level(logging.WARNING, logger="familiar.campaign"):
        refresh_campaign(d, store=_store(tmp_path), llm=llm)
    body = campaign.read_body(d / CAMPAIGN_FILE)
    assert len(body) <= CAMPAIGN_CAP and body.endswith("on and on.\n")
    assert any("cap" in r.message for r in caplog.records if r.levelno == logging.WARNING)


def test_it_is_indexed_in_sessions_under_its_own_source_just_above_a_summary(tmp_path):
    from familiar_brain.transcript import DELIBERATE_WEIGHT
    d = _sessions(tmp_path, **{"20260916-a": S1})
    store, llm = _store(tmp_path), MergeLLM()
    refresh_campaign(d, store=store, llm=llm)
    rows = _rows(store)
    # Above a session summary's 1.5, below a deliberate note (deviation P19).
    assert rows and all(w == CAMPAIGN_WEIGHT == 1.6 for _, _, w in rows)
    assert 1.5 < CAMPAIGN_WEIGHT < DELIBERATE_WEIGHT
    assert store.db.execute("SELECT DISTINCT corpus FROM chunks").fetchall() == [("sessions",)]
    # Its own chunk per heading, so "Open threads" is retrievable on its own.
    titles = [t for t, _, _ in rows]
    assert any("Open threads" in t for t in titles) and any("Story so far" in t for t in titles)
    assert all("folded" not in text for _, text, _ in rows)  # the bookkeeping line is not indexed
    _sessions(tmp_path, **{"20260923-b": S2})
    refresh_campaign(d, store=store, llm=llm)
    assert len(_rows(store)) == len(rows)  # replaced, not doubled
    assert all("Thread 1" not in text for _, text, _ in _rows(store))


def test_the_memory_is_scrubbed_of_players(tmp_path):
    d = _sessions(tmp_path, **{"20260916-a": S1})
    store = _store(tmp_path)
    llm = MergeLLM(reply=lambda m: memory("1") + "- James found the key\n")
    refresh_campaign(d, store=store, llm=llm, players=["James"])
    assert "James" not in (d / CAMPAIGN_FILE).read_text()
    assert all("James" not in text for _, text, _ in _rows(store))


def test_a_failed_merge_writes_nothing_and_stays_due(tmp_path):
    d = _sessions(tmp_path, **{"20260916-a": S1})
    with pytest.raises(RuntimeError):
        refresh_campaign(d, store=_store(tmp_path), llm=MergeLLM(fail=RuntimeError("timed out")))
    assert not (d / CAMPAIGN_FILE).exists() and campaign_due(d)
    with pytest.raises(ValueError):
        refresh_campaign(d, store=_store(tmp_path), llm=MergeLLM(reply=lambda m: " "))
    assert not (d / CAMPAIGN_FILE).exists()


def test_purging_a_session_drops_the_memory_now_and_the_rebuild_forgets_it(tmp_path):
    d = _sessions(tmp_path, **{"20260916-a": S1})
    store, llm = _store(tmp_path), MergeLLM()
    refresh_campaign(d, store=store, llm=llm)
    _sessions(tmp_path, **{"20260923-b": S2})
    refresh_campaign(d, store=store, llm=llm)
    assert purge_session(d, "20260916-a", store=store)
    # Gone at once, with no model call, so nothing the purged session said is retrievable.
    assert not (d / CAMPAIGN_FILE).exists() and not (d / PREV_FILE).exists() and _rows(store) == []
    assert campaign_due(d)
    llm.calls.clear()
    assert refresh_campaign(d, store=store, llm=llm) == 1
    assert S2 in llm.user() and S1 not in llm.user() and "nothing yet" in llm.user()


def test_purging_a_session_the_memory_never_held_leaves_it(tmp_path):
    d = _sessions(tmp_path, **{"20260916-a": S1})
    store, llm = _store(tmp_path), MergeLLM()
    refresh_campaign(d, store=store, llm=llm)
    (d / "20260930-x.jsonl").write_text("{}\n", encoding="utf-8")
    purge_session(d, "20260930-x", store=store)
    assert (d / CAMPAIGN_FILE).exists() and _rows(store)


def test_re_summarising_or_learning_an_older_session_late_rebuilds_in_date_order(tmp_path):
    d = _sessions(tmp_path, **{"20260916-a": S1, "20260923-b": S2})
    store, llm = _store(tmp_path), MergeLLM()
    refresh_campaign(d, store=store, llm=llm)
    (d / "20260916-a.summary.md").write_text(S1 + "- And the mayor lied\n", encoding="utf-8")
    llm.calls.clear()
    assert refresh_campaign(d, store=store, llm=llm) == 2
    assert "nothing yet" in llm.user(0) and "mayor lied" in llm.user(0) and S2 in llm.user(1)
    # A session older than the last one folded, learnt late: rebuilt, oldest first.
    _sessions(tmp_path, **{"20260910-z": "## Scenes\n- The ferry\n"})
    llm.calls.clear()
    assert refresh_campaign(d, store=store, llm=llm) == 3
    assert "The ferry" in llm.user(0)


def test_a_session_starting_mid_rebuild_stops_it_between_folds(tmp_path):
    d = _sessions(tmp_path, **{"20260916-a": S1, "20260923-b": S2})
    llm = MergeLLM()
    calls = iter([False, True])
    assert refresh_campaign(d, store=_store(tmp_path), llm=llm, stop=lambda: next(calls)) == 1
    assert campaign_due(d)  # the second session waits for the next idle moment
    assert refresh_campaign(d, store=_store(tmp_path), llm=llm) == 1 and S2 in llm.user()


def test_what_to_do_next_reaches_the_open_threads(tmp_path):
    """build_context retrieves by similarity, per corpus. The memory's Open threads are a chunk
    of their own, titled so a "what now?" question finds them among the session chunks."""
    d = _sessions(tmp_path, **{"20260916-a": S1})
    store = _store(tmp_path)
    store.add_chunks([Chunk("sessions", f"2026091{i}-x.jsonl", f"Session 2026091{i}-x transcript", t,
                            {"session": f"2026091{i}-x"}, 1.0) for i, t in enumerate([
        "Should we go now? I think we should rest first, the cleric is out of spells.",
        "Rocky, do you remember the name of the ferryman? He wanted silver.",
        "The bell tower is locked and the chapel crypt smells of lamp oil.",
        "We sold the silver candlesticks in Saltmere for forty gold."])])
    thread = "Find the bell-ringer before the next new moon"
    llm = MergeLLM(reply=lambda m: memory("1").replace("- Thread 1", f"- {thread}")
                   + "".join(f"- Ledger entry {i}: rope, rations, a lantern and oil\n" for i in range(40)))
    refresh_campaign(d, store=store, llm=llm)
    ctx = build_context(store, "Rocky, what should we do next?", chunks_per_corpus=3, max_chars=4800)
    assert thread in ctx
    assert "Ledger entry 39" not in ctx  # the whole memory is not in every prompt


def test_the_learner_folds_each_new_summary_and_rebuilds_after_a_purge(tmp_path):
    d = tmp_path / "sessions"; d.mkdir()
    for sid, text in (("20260916-a", "The well is poisoned."), ("20260923-b", "The crypt is cold.")):
        (d / f"{sid}.jsonl").write_text(json.dumps({"ts": 1, "text": text, "decision": "IGNORE",
                                                    "spoke": False}) + "\n", encoding="utf-8")

    class LLM:
        calls = []

        def complete(self, messages, max_tokens=1200):
            self.calls.append(messages[0]["content"])
            return memory("x") if messages[0]["content"] == CAMPAIGN_PROMPT else S1

    store, llm = _store(tmp_path), LLM()
    state = {"active": None}
    lr = Learner(d, store=store, llm=llm, players_path=tmp_path / "players.txt",
                 active=lambda: state["active"], spawn=lambda fn: fn())
    lr.catch_up()
    assert llm.calls.count(CAMPAIGN_PROMPT) == 2 and not campaign_due(d)
    purge_session(d, "20260916-a", store=store)
    state["active"] = "20261002-live"
    assert lr.kick() is False  # never while a session is going
    state["active"] = None
    llm.calls.clear()
    assert lr.kick() is True
    assert llm.calls == [CAMPAIGN_PROMPT] and not campaign_due(d)


def test_learn_all_folds_the_campaign_memory_too(tmp_path):
    import io
    d = _sessions(tmp_path, **{"20260916-a": S1})
    out = io.StringIO()
    assert learn_all(d, store=_store(tmp_path), llm=MergeLLM(), players_path=tmp_path / "p.txt", out=out) == 0
    assert (d / CAMPAIGN_FILE).exists() and "campaign memory" in out.getvalue()


def test_an_over_cap_memory_keeps_every_heading_and_open_thread(tmp_path):
    """The cap cut from the end, and Story so far comes first and grows fastest, so the first
    over-long reply lost NPCs, Places, Open threads, Loot and Decisions, and the next merge
    built on what was left. Story so far now gives up its oldest lines instead."""
    d = _sessions(tmp_path, **{"20260916-a": S1})
    story = "".join(f"- Chapter {i}: the party walked a long and winding road.\n" for i in range(110))
    reply = ("## Story so far\n" + story + "## NPCs\n- Odo the ferryman\n## Places\n- Saltmere\n"
             "## Open threads\n- Who poisoned the well?\n## Loot and resources\n- A black lotus seal\n"
             "## Decisions that matter\n- Spared the courier\n")
    assert len(reply) > CAMPAIGN_CAP
    refresh_campaign(d, store=_store(tmp_path), llm=MergeLLM(reply=lambda m: reply))
    body = campaign.read_body(d / CAMPAIGN_FILE)
    assert len(body) <= CAMPAIGN_CAP
    for kept in ("## Open threads\n- Who poisoned the well?", "Odo the ferryman", "Saltmere",
                 "A black lotus seal", "Spared the courier", "Chapter 109"):
        assert kept in body, kept
    assert "Chapter 0:" not in body


def test_a_session_summary_fact_is_still_retrievable_once_a_campaign_memory_exists(tmp_path):
    """At weight 2.0 against a summary's 1.5, the campaign chunks took all three sessions
    slots for almost any question, empty '- none' sections included, so a fact only a
    session summary kept (the seal on a letter) never reached the prompt or recall. Under
    the BGE embedder at 1.6 the summary's Loot, NPCs met and Memorable lines chunks each
    reach the top three for the questions only they answer, and recap questions still go to
    the memory (measured 2026-10-02, deviation P19)."""
    d = _sessions(tmp_path, **{"20260916-a": S1})
    store = _store(tmp_path)
    store.add_chunks([Chunk("sessions", "20260916-a.summary.md", "Session 20260916-a: Loot",
                            "- A sealed letter with a black lotus seal", {"session": "20260916-a"}, 1.5)])
    mem = ("## Story so far\n- The party travelled to Saltmere and took a letter from a courier\n"
           "## NPCs\n- Mirela, a courier\n## Places\n- Saltmere, a fishing town\n"
           "## Open threads\n- Who is the letter for?\n## Loot and resources\n- A sealed letter\n"
           "## Decisions that matter\n- none\n")
    refresh_campaign(d, store=store, llm=MergeLLM(reply=lambda m: mem))
    assert not any("Decisions" in title for title, _, _ in _rows(store))  # a '- none' section
    hits = store.search("what was the seal on the letter?", corpus="sessions", k=3)
    assert any(h.source == "20260916-a.summary.md" for h in hits)
