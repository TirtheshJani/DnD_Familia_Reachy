from __future__ import annotations

from pathlib import Path

from familiar_brain.indexing import index_notes
from familiar_brain.persona import load_persona
from familiar_brain.retrieval import Chunk, Store
from familiar_brain.tools import lookup_rule, recall
from tests.fakes.embedder import FakeEmbedder

ROOT = Path(__file__).resolve().parent.parent


def test_nephria_update_files_are_copied_to_repo_targets():
    expected_pairs = [
        ("nephria-update/notes/10-brokk-trenevain.md", "data/notes/10-brokk-trenevain.md"),
        ("nephria-update/notes/11-roster-additions.md", "data/notes/11-roster-additions.md"),
        ("nephria-update/notes/12-recap-old-town-and-jade-halls.md", "data/notes/12-recap-old-town-and-jade-halls.md"),
        ("nephria-update/homebrew/rocky-familiar.md", "data/notes/homebrew/rocky-familiar.md"),
        ("nephria-update/homebrew/brokk-sheet.md", "data/notes/homebrew/brokk-sheet.md"),
        ("nephria-update/homebrew/house-rules-2026-10-02.md", "data/notes/homebrew/house-rules-2026-10-02.md"),
        ("nephria-update/docs/rocky-persona-facts.md", "docs/rocky-persona-facts.md"),
    ]
    for src_rel, dst_rel in expected_pairs:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        assert dst.exists(), f"Missing copied file: {dst_rel}"
        assert dst.read_text(encoding="utf-8") == src.read_text(encoding="utf-8"), dst_rel


def test_sep11_mock_sessions_purged_and_sep16_preserved(tmp_path: Path):
    import json
    import os
    import re
    import time
    from familiar_brain.summariser import apply_retention, load_players

    sessions_dir = ROOT / "data" / "sessions"
    assert sessions_dir.exists()
    sep11 = list(sessions_dir.glob("20260911-*"))
    assert sep11 == [], f"Expected Sep 11 mock logs to be removed, found: {[p.name for p in sep11]}"

    sep16_raw = sessions_dir / "20260916-182232-0189cc.jsonl"
    sep16_sum = sessions_dir / "20260916-182232-0189cc.summary.md"
    assert (sep16_raw.exists() and sep16_raw.stat().st_size > 0) or (
        sep16_sum.exists() and sep16_sum.stat().st_size > 0
    ), "Expected Sep 16 session artifact (.jsonl or .summary.md) to be preserved"

    players = load_players(ROOT / "config" / "players.txt")
    assert {"James", "Fiona", "Jenna", "Ken", "Riley"} <= set(players)
    pat = re.compile(r"\b(" + "|".join(re.escape(p) for p in players) + r")\b", re.IGNORECASE)
    for md_file in (sep16_sum, sessions_dir / "campaign-memory.md"):
        if md_file.exists():
            assert not pat.search(md_file.read_text(encoding="utf-8")), f"Real player name leaked in {md_file.name}"

    # Guard: apply_retention(..., retention_days=0) preserves existing summarized .jsonl files
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    raw = sdir / "20260916-182232-0189cc.jsonl"
    raw.write_text(json.dumps({"ts": 1, "text": "The jade hall gleams.", "decision": "IGNORE", "spoke": False}) + "\n", encoding="utf-8")
    raw.with_suffix(".summary.md").write_text("## Scenes\n- Jade hall\n", encoding="utf-8")
    old = time.time() - 90 * 86400
    os.utime(raw, (old, old))
    store = Store(tmp_path / "ret.db", embedder=FakeEmbedder())
    try:
        assert apply_retention(sdir, store=store, retention_days=0) == []
        assert raw.exists()
    finally:
        store.close()


def test_notes_policy_documents_update_and_srd_521():
    policy = (ROOT / "docs" / "notes-policy.md").read_text(encoding="utf-8")
    assert "nephria-update/notes/*.md" in policy
    assert "SRD_CC_v5.2.1.pdf" in policy
    assert "index rules data/rules/SRD_CC_v5.2.1.pdf --edition 2024" in policy


def test_persona_yaml_matches_rocky_aliases_and_gestures_if_present():
    p = ROOT / "config" / "persona.yaml"
    if not p.exists():
        return
    per = load_persona(p)
    assert per.name == "Rocky"
    assert set(["rocky", "rocki", "rockie"]).issubset(set(per.aliases))
    assert "Construct" in per.manner and "Brokk" in per.manner
    for mood in ("attentive", "amused", "alarmed", "thoughtful", "proud", "sad", "scared", "asleep", "awake", "cheer"):
        assert mood in per.gestures


def test_index_notes_and_retrieve_nephria_campaign_and_rules_priority(tmp_path: Path):
    s = Store(tmp_path / "test_r3.db", embedder=FakeEmbedder())
    res = index_notes(s, ROOT / "data" / "notes")
    assert res["indexed"] >= 16
    assert "10-brokk-trenevain.md" in s.sources("notes")
    assert "11-roster-additions.md" in s.sources("notes")
    assert "12-recap-old-town-and-jade-halls.md" in s.sources("notes")
    assert "homebrew/rocky-familiar.md" in s.sources("homebrew")
    assert "homebrew/brokk-sheet.md" in s.sources("homebrew")
    assert "homebrew/house-rules-2026-10-02.md" in s.sources("homebrew")

    recall.bind(s)
    brokk_out = recall.run({"query": "Brokk Trenevain Gloamhaunt Vlad", "scope": "notes"})
    assert "Brokk" in brokk_out or "Trenevain" in brokk_out

    recap_out = recall.run({
        "query": "The Grand Hall They woke as if from a dream on wet worked jade in total darkness centimetre of water",
        "scope": "notes",
    })
    assert "Grand Hall" in recap_out or "worked jade" in recap_out

    # Add synthetic 2014 and 2024 rules alongside indexed homebrew to verify precedence
    s.add_chunks([
        Chunk("rules", "srd51.pdf", "Find Familiar", "2014 familiar rule.", {"edition": "2014"}),
        Chunk("rules", "SRD_CC_v5.2.1.pdf", "Find Familiar", "2024 familiar rule.", {"edition": "2024"}),
    ])
    lookup_rule.bind(s)
    rocky_rule = lookup_rule.run({"query": "Rocky Spelltouch familiar"})
    assert rocky_rule.startswith("House rule")
    assert "2024 rules" in rocky_rule and "2014 rules" in rocky_rule
    assert rocky_rule.index("2024 rules") < rocky_rule.index("2014 rules")
