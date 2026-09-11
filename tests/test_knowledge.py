from familiar_brain.knowledge import build_context
from familiar_brain.retrieval import Chunk, Store
from tests.fakes.embedder import FakeEmbedder


def test_build_context_caps_per_corpus_and_total(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks([Chunk("rules", "r.pdf", f"Rule {i}", "grapple " * 200, {"edition": "2014"}) for i in range(6)])
    s.add_chunks([Chunk("notes", "n.md", "Note", "grapple in the notes", {})])
    ctx = build_context(s, "how does grapple work", chunks_per_corpus=3, max_chars=1500)
    assert ctx.startswith("What you know:") and ctx.count("Rule ") <= 3 and len(ctx) <= 1600
    assert "Note" in ctx or len(ctx) >= 1400  # notes included unless the cap was hit first


def test_build_context_empty_store_returns_empty_string(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    assert build_context(s, "anything", chunks_per_corpus=3, max_chars=1000) == ""


def test_build_context_includes_homebrew_before_rules(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks([Chunk("rules", "srd.pdf", "Critical Hits", "crit doubles dice", {"edition": "2014"}),
                  Chunk("homebrew", "homebrew/house-rules.md", "Injuries", "crit may become an injury", {})])
    ctx = build_context(s, "what happens on a crit", chunks_per_corpus=3, max_chars=2000)
    assert ctx.index("[homebrew]") < ctx.index("[rules]")


def test_build_context_names_the_edition_of_a_rules_hit(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks([Chunk("rules", "srd52.pdf", "Grappling", "grapple text", {"edition": "2024"})])
    assert "(2024 rules)" in build_context(s, "grapple", chunks_per_corpus=3, max_chars=2000)
