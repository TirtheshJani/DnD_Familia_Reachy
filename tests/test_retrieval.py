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
