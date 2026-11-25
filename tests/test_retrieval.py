from pathlib import Path
from familiar_brain.retrieval import Store, Chunk
from tests.fakes.embedder import FakeEmbedder


def make(tmp: Path) -> Store:
    return Store(tmp / "k.db", embedder=FakeEmbedder())


def test_add_and_search_returns_best_match_with_meta(tmp_path):
    s = make(tmp_path)
    s.add_chunks([
        Chunk(corpus="rules", source="srd51.pdf", title="Grappling", text="When you want to grab a creature you can use the Attack action to make a special melee attack, a grapple.", meta={"edition": "2014"}),
        Chunk(corpus="rules", source="srd51.pdf", title="Fireball", text="A bright streak flashes from your pointing finger to a point you choose.", meta={"edition": "2014"}),
        Chunk(corpus="notes", source="pc.md", title="Backstory", text="My character grew up on the docks of Nephria.", meta={}),
    ])
    hits = s.search("how does grapple work", corpus="rules", k=1)
    assert hits[0].title == "Grappling" and hits[0].meta["edition"] == "2014" and hits[0].score > 0


def test_search_is_scoped_to_corpus(tmp_path):
    s = make(tmp_path)
    s.add_chunks([Chunk(corpus="notes", source="a.md", title="A", text="grapple grapple grapple", meta={})])
    assert s.search("grapple", corpus="rules", k=3) == []


def test_delete_source_removes_its_chunks_only(tmp_path):
    s = make(tmp_path)
    s.add_chunks([
        Chunk(corpus="notes", source="a.md", title="A", text="alpha", meta={}),
        Chunk(corpus="notes", source="b.md", title="B", text="beta", meta={}),
    ])
    s.delete_source("a.md")
    assert [h.title for h in s.search("alpha beta", corpus="notes", k=5)] == ["B"]


def test_weight_scales_score(tmp_path):
    s = make(tmp_path)
    s.add_chunks([
        Chunk(corpus="sessions", source="s1.summary.md", title="Summary", text="the barkeep owes gold", meta={}, weight=1.5),
        Chunk(corpus="sessions", source="s1.jsonl", title="Raw", text="the barkeep owes gold", meta={}, weight=1.0),
    ])
    hits = s.search("barkeep gold", corpus="sessions", k=2)
    assert hits[0].title == "Summary" and hits[0].score > hits[1].score


def test_search_still_finds_a_small_corpus_beside_a_large_one(tmp_path):
    """Corpus scoping has to happen inside the KNN query, not after it.

    Pulling a fixed candidate set and then filtering by corpus loses the notes
    entirely once the rules corpus dominates, which is exactly the shape of the real
    store: two SRDs, thousands of chunks, against a handful of campaign notes that
    share their vocabulary. Observed with the post-filter form: 0 hits.
    """
    import random
    random.seed(7)
    s = make(tmp_path)
    vocab = ["gold", "barkeep", "tavern", "inn", "coin", "party", "owes", "silver", "keeper", "eel"]
    s.add_chunks([Chunk("rules", "srd.pdf", f"Rule {i}", " ".join(random.choices(vocab, k=12)),
                        {"edition": "2014"}) for i in range(2000)])
    s.add_chunks([Chunk("notes", "npcs.md", "Barkeep",
                        "The barkeep at the Gilded Eel owes the party forty gold.", {})])
    hits = s.search("barkeep gold", corpus="notes", k=3)
    assert [h.title for h in hits] == ["Barkeep"]


def test_search_corpora_embeds_once_and_matches_sqlite_vec_equivalence(tmp_path):
    import json
    from familiar_brain.knowledge import build_context
    from familiar_brain.retrieval import Hit

    class CountingEmbedder(FakeEmbedder):
        def __init__(self):
            self.calls = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return super().embed(texts)

    emb = CountingEmbedder()
    s = Store(tmp_path / "k.db", embedder=emb)
    s.add_chunks([
        Chunk("homebrew", "hb.md", "Injuries", "Critical hits cause lingering injuries.", {}, 1.0),
        Chunk("rules", "srd.pdf", "Grappling", "You can grapple a creature using an Attack action.", {"edition": "2024"}, 1.0),
        Chunk("rules", "srd.pdf", "Fireball", "A bright streak flashes from your pointing finger.", {"edition": "2024"}, 1.0),
        Chunk("notes", "brokk.md", "Brokk", "Brokk Trenevain is an artificer bound to Rocky.", {}, 1.0),
        Chunk("sessions", "s1.summary.md", "Scenes", "The party fought near the cistern.", {}, 1.5),
        Chunk("sessions", "campaign-memory.md", "Story", "Brokk examined the lingering injury.", {}, 1.6),
    ])

    emb.calls = 0
    ctx = build_context(s, "Brokk lingering injury grapple", chunks_per_corpus=2, max_chars=4800)
    assert emb.calls == 1, f"Expected build_context to embed query once, got {emb.calls}"
    assert "Injuries" in ctx and "Brokk" in ctx

    q = "Brokk lingering injury grapple"
    q_vec = s.embedder.embed([q])[0]
    for corpus in ("homebrew", "rules", "notes", "sessions"):
        fast = s.search(q, corpus=corpus, k=2, query_vec=q_vec)
        rows = s._knn(q_vec, 2, corpus)
        slow = [Hit(r[0], r[1], r[2], r[3], json.loads(r[4]), score=(1.0 - r[6] / 2.0) * r[5]) for r in rows]
        slow.sort(key=lambda h: -h.score)
        slow = slow[:2]
        assert [(h.source, h.title, round(h.score, 5)) for h in fast] == [
            (h.source, h.title, round(h.score, 5)) for h in slow
        ]


def test_store_cache_invalidates_on_add_replace_and_delete_pieces(tmp_path):
    s = make(tmp_path)
    s.add_chunks([
        Chunk("sessions", "s1.jsonl", "Part 0", "goblin ambush", {"piece": 0}, 1.0),
        Chunk("sessions", "s1.jsonl", "Part 1", "secret tunnel", {"piece": 1}, 1.0),
    ])
    assert len(s.search("secret tunnel", corpus="sessions", k=5)) == 2
    s.delete_pieces("s1.jsonl", 1)
    hits_after_piece = s.search("secret tunnel", corpus="sessions", k=5)
    assert [h.title for h in hits_after_piece] == ["Part 0"]
    s.add_chunks([Chunk("sessions", "s1.jsonl", "Part 0b", "dragon hoard", {"piece": 0}, 1.0)], replace=("s1.jsonl",))
    hits_after_replace = s.search("dragon hoard", corpus="sessions", k=5)
    assert [h.title for h in hits_after_replace] == ["Part 0b"]


def test_cli_module_entrypoint_fails_nonzero_on_missing_pdf(tmp_path):
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "familiar_brain.cli", "index", "rules", str(tmp_path / "missing.pdf"), "--edition", "2024"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode != 0
    assert "not found" in (proc.stderr + proc.stdout).lower()
