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
