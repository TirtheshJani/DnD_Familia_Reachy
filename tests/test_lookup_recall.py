from pathlib import Path

from familiar_brain.retrieval import Store, Chunk
from familiar_brain.tools import lookup_rule, recall
from tests.fakes.embedder import FakeEmbedder


def store(tmp: Path) -> Store:
    s = Store(tmp / "k.db", embedder=FakeEmbedder())
    s.add_chunks([
        Chunk("rules", "srd51.pdf", "Grappling", "Grapple 2014 text about grabbing a creature.", {"edition": "2014"}),
        Chunk("rules", "srd52.pdf", "Grappling", "Grapple 2024 text, unarmed strike grapple option.", {"edition": "2024"}),
        Chunk("notes", "npcs.md", "Barkeep", "The barkeep owes the party forty gold.", {}),
        Chunk("sessions", "s1.summary.md", "Session 1", "The party met the barkeep at the Gilded Eel.", {}, 1.5),
    ])
    return s


def test_lookup_rule_returns_one_per_edition_and_names_it(tmp_path):
    lookup_rule.bind(store(tmp_path))
    out = lookup_rule.run({"query": "grapple a creature"})
    assert "2014 rules" in out and "2024 rules" in out and out.count("Grappling") == 2


def test_lookup_rule_empty_store_is_graceful(tmp_path):
    lookup_rule.bind(Store(tmp_path / "e.db", embedder=FakeEmbedder()))
    assert "nothing" in lookup_rule.run({"query": "x"}).lower()


def test_recall_scopes(tmp_path):
    s = store(tmp_path); recall.bind(s)
    assert "forty gold" in recall.run({"query": "barkeep gold", "scope": "notes"})
    assert "Gilded Eel" in recall.run({"query": "barkeep", "scope": "sessions"})
    both = recall.run({"query": "barkeep"})
    assert "forty gold" in both and "Gilded Eel" in both


def test_lookup_rule_prefers_house_rule(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks([
        Chunk("homebrew", "homebrew/house-rules.md", "Sanity skills", "Revelation, Weird, Grounding, Entropic Paradox are Sanity skills.", {}),
        Chunk("rules", "srd51.pdf", "Wisdom", "Wisdom measures perception and insight.", {"edition": "2014"}),
    ])
    lookup_rule.bind(s)
    out = lookup_rule.run({"query": "how do Sanity skills work"})
    assert out.startswith("House rule") and "Sanity skills" in out
    assert "2014 rules" in out  # SRD still follows
