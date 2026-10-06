"""The live indexer: the transcript so far, searchable mid-session, by embedding alone.

Driven synchronously: `spawn` is a no-op so no worker thread runs, and every tick is
given its clock reading, so no test sleeps.
"""
import pytest

from familiar_brain.live_index import BATCH_AGE_S, BATCH_LINES, LiveIndexer
from familiar_brain.retrieval import Store
from familiar_brain.session import SessionLog
from familiar_brain.summariser import summarise_session
from tests.fakes.embedder import FakeEmbedder


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        self.embedded: list[str] = []
        self.fail_next = False

    def embed(self, texts):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("embedder fell over")
        self.embedded += texts
        return super().embed(texts)


@pytest.fixture
def live(tmp_path):
    emb = CountingEmbedder()
    store = Store(tmp_path / "k.db", embedder=emb)
    log = SessionLog(tmp_path / "sessions")
    log.start()
    idx = LiveIndexer(store, spawn=lambda fn: None)
    idx.begin(log.path)
    return idx, log, store, emb


def _say(log, text, tag=None):
    log.append(text=text, decision="IGNORE", spoke=False, tag=tag)


def _rows(store, source):
    return store.db.execute("SELECT text, weight FROM chunks WHERE source = ? ORDER BY id", (source,)).fetchall()


def test_nothing_is_embedded_until_a_batch_is_due(live):
    idx, log, store, emb = live
    _say(log, "The innkeeper says the well is poisoned.")
    idx.tick(0.0)
    idx.tick(BATCH_AGE_S - 1)
    assert emb.embedded == []
    idx.tick(BATCH_AGE_S)
    hits = store.search("innkeeper well poisoned", corpus="sessions", k=3)
    assert hits and "poisoned" in hits[0].text and hits[0].source == log.path.name


def test_a_full_batch_is_indexed_without_waiting(live):
    idx, log, store, _ = live
    for i in range(BATCH_LINES):
        _say(log, f"Line {i} about the smuggler's tunnel.")
    idx.tick(0.0)
    assert store.search("smuggler tunnel", corpus="sessions", k=1)


def test_a_deliberate_note_is_indexed_on_the_next_tick_and_outranks_the_table(live):
    idx, log, store, _ = live
    _say(log, "The innkeeper is called Bram.")
    _say(log, "The innkeeper is called Bram.", tag="deliberate")
    idx.tick(0.0)
    hits = store.search("innkeeper Bram", corpus="sessions", k=2)
    assert [h.meta.get("deliberate") for h in hits] == [True, None]
    assert _rows(store, log.path.name)[0][1] == 2.0


def test_revisions_are_embedded_once_in_their_final_form(live):
    idx, log, store, _ = live
    _say(log, "Would you like to pick up I can see that?")
    _say(log, "Would you like to piss up? I can't see it.")
    idx.tick(0.0); idx.tick(BATCH_AGE_S)
    assert _rows(store, log.path.name) == [("Would you like to piss up? I can't see it.", 1.0)]


def test_a_revision_after_a_flush_replaces_the_indexed_tail(live):
    idx, log, store, _ = live
    _say(log, "The bridge is out, we go round.")
    idx.tick(0.0); idx.tick(BATCH_AGE_S)
    _say(log, "The bridge is out, we go round by the ford.")
    idx.tick(100.0); idx.tick(100.0 + BATCH_AGE_S)
    assert _rows(store, log.path.name) == [("The bridge is out, we go round by the ford.", 1.0)]


def test_only_the_changed_tail_is_embedded_again(live):
    idx, log, store, emb = live
    for i in range(80):
        # Words of their own, or every line would read as a revision of the one before.
        _say(log, " ".join(f"w{i}x{j}" for j in range(10)) + ".")
    idx.tick(0.0)
    assert len(_rows(store, log.path.name)) > 3
    before = len(emb.embedded)
    _say(log, "And then the wizard sneezed.")
    idx.tick(1.0); idx.tick(1.0 + BATCH_AGE_S)
    assert len(emb.embedded) - before == 1


def test_end_indexes_the_rest_and_nothing_after(live):
    idx, log, store, emb = live
    _say(log, "The cult meets under the mill.")
    idx.end()
    assert store.search("cult mill", corpus="sessions", k=1)
    n = len(emb.embedded)
    _say(log, "A line written after the end.")
    idx.tick(1e6)
    assert len(emb.embedded) == n


def test_a_failed_embed_is_tried_again_on_the_next_tick(live):
    idx, log, store, emb = live
    _say(log, "Bram owes us ten gold.", tag="deliberate")
    emb.fail_next = True
    idx.tick(0.0)
    assert _rows(store, log.path.name) == []
    idx.tick(5.0)
    assert _rows(store, log.path.name) == [("Bram owes us ten gold.", 2.0)]


def test_a_half_written_last_line_waits_for_the_rest(live):
    idx, log, store, _ = live
    with log.path.open("a", encoding="utf-8") as f:
        f.write('{"ts": 1, "text": "The vault opens at dusk.", "decision": "IGNORE", "spoke": fal')
    idx.tick(0.0); idx.tick(BATCH_AGE_S)
    assert _rows(store, log.path.name) == []
    with log.path.open("a", encoding="utf-8") as f:
        f.write('se}\n')
    idx.tick(BATCH_AGE_S + 1); idx.tick(2 * BATCH_AGE_S + 1)
    assert _rows(store, log.path.name) == [("The vault opens at dusk.", 1.0)]


def test_summarising_replaces_the_live_chunks_and_keeps_the_notes(live):
    idx, log, store, _ = live
    _say(log, "The innkeeper is called Bram.")
    _say(log, "Bram owes us ten gold.", tag="deliberate")
    idx.tick(0.0)
    idx.end()
    path = log.end()

    class LLM:
        def complete(self, messages, max_tokens=1200):
            return "## Scenes\n- The inn\n"

    out = summarise_session(path, store=store, llm=LLM())
    assert _rows(store, path.name) == [("The innkeeper is called Bram.", 1.0)]
    assert "Bram owes us ten gold." in out.read_text(encoding="utf-8")
    note = store.search("Bram owes ten gold", corpus="sessions", k=1)[0]
    assert note.source == out.name
    assert store.db.execute("SELECT weight FROM chunks WHERE title LIKE '%Deliberate notes'").fetchall() == [(2.0,)]
