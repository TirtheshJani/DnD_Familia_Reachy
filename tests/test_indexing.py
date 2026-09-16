import os
import time
from pathlib import Path

from familiar_brain.indexing import index_notes
from familiar_brain.retrieval import Store
from tests.fakes.embedder import FakeEmbedder


def _from_an_earlier_sitting(*notes: Path) -> None:
    """Backdate notes a minute, as notes written before the session would be.

    On Windows both the NTFS mtime and SQLite's created_at come from the tick-based
    system clock, but SQLite cuts its reading to whole milliseconds. A note written in
    the same tick as the first pass (15.625 ms by default) therefore looks up to 1 ms
    newer than its own chunks, and the second pass re-indexes it once. Any test that
    writes notes and then expects them skipped fails intermittently without this. The
    error only runs that way: an edit saved in a later tick carries the later mtime.
    """
    earlier = time.time() - 60
    for note in notes:
        os.utime(note, (earlier, earlier))


def test_index_notes_skips_unchanged_and_reindexes_changed(tmp_path: Path):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "pc.md").write_text("# PC\nDocks of Nephria.\n")
    (notes / "npcs.md").write_text("# Barkeep\nOwes gold.\n")
    _from_an_earlier_sitting(notes / "pc.md", notes / "npcs.md")
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    assert index_notes(store, notes) == {"indexed": 2, "skipped": 0}
    assert index_notes(store, notes) == {"indexed": 0, "skipped": 2}
    (notes / "pc.md").write_text("# PC\nNow lives uptown.\n")
    os.utime(notes / "pc.md", (time.time() + 5, time.time() + 5))
    assert index_notes(store, notes) == {"indexed": 1, "skipped": 1}
    assert store.search("uptown", corpus="notes", k=1)[0].text.startswith("Now lives")
    assert not any("Docks" in h.text for h in store.search("docks", corpus="notes", k=5))  # old chunk gone


def test_homebrew_subfolder_goes_to_homebrew_corpus_and_txt_is_indexed(tmp_path: Path):
    notes = tmp_path / "notes"; (notes / "homebrew").mkdir(parents=True)
    (notes / "homebrew" / "house-rules.md").write_text(
        "# Injuries\nOn a crit you may take an injury instead of double damage.\n")
    (notes / "lore.txt").write_text("The Cobalt River separates Nephria from Xeria.\n")
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    assert index_notes(store, notes) == {"indexed": 2, "skipped": 0}
    assert store.search("injury crit", corpus="homebrew", k=1)[0].title == "Injuries"
    assert store.search("Cobalt River", corpus="notes", k=1)[0].source == "lore.txt"
    # a house rule must not also land in the notes corpus
    assert all(h.title != "Injuries" for h in store.search("injury", corpus="notes", k=3))


def test_homebrew_files_are_skipped_on_a_second_pass_too(tmp_path: Path):
    """sources() is per corpus, so a homebrew file is invisible to sources('notes').
    Without merging both, every reindex would re-chunk the house rules."""
    notes = tmp_path / "notes"; (notes / "homebrew").mkdir(parents=True)
    (notes / "homebrew" / "house-rules.md").write_text("# Injuries\nCrit may become an injury.\n")
    (notes / "pc.md").write_text("# PC\nDocks of Nephria.\n")
    _from_an_earlier_sitting(notes / "homebrew" / "house-rules.md", notes / "pc.md")
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    assert index_notes(store, notes) == {"indexed": 2, "skipped": 0}
    assert index_notes(store, notes) == {"indexed": 0, "skipped": 2}


class _SpyEmbedder:
    dim = 64

    def __init__(self):
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.0] * self.dim for _ in texts]


class _BrokenEmbedder:
    dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("no model on disk")


def test_warm_store_loads_the_embedder_up_front(tmp_path: Path):
    """Measured on this machine: 11.9 s for the first embed, 21 ms warm. Paid on the
    first addressed turn that would blow the 2.5 s latency ceiling and expire the 8 s
    follow-up window before the player's follow-up ever arrived."""
    from familiar_brain.indexing import warm_store
    spy = _SpyEmbedder()
    warm_store(Store(tmp_path / "k.db", embedder=spy))
    assert spy.calls == 1


def test_warm_store_survives_an_embedder_that_cannot_load(tmp_path: Path):
    """A missing model must not stop the brain from starting. Retrieval degrades to
    nothing, which app.py already handles, and the table still gets its familiar."""
    from familiar_brain.indexing import warm_store
    warm_store(Store(tmp_path / "k.db", embedder=_BrokenEmbedder()))  # must not raise
