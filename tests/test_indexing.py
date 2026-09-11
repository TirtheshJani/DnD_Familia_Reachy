from pathlib import Path

from familiar_brain.indexing import index_notes
from familiar_brain.retrieval import Store
from tests.fakes.embedder import FakeEmbedder


def test_index_notes_skips_unchanged_and_reindexes_changed(tmp_path: Path):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "pc.md").write_text("# PC\nDocks of Nephria.\n")
    (notes / "npcs.md").write_text("# Barkeep\nOwes gold.\n")
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    assert index_notes(store, notes) == {"indexed": 2, "skipped": 0}
    assert index_notes(store, notes) == {"indexed": 0, "skipped": 2}
    import os, time
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
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    assert index_notes(store, notes) == {"indexed": 2, "skipped": 0}
    assert index_notes(store, notes) == {"indexed": 0, "skipped": 2}
