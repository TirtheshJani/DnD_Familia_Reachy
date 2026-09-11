import json
import time
from pathlib import Path

from familiar_brain.retrieval import Store
from familiar_brain.summariser import (
    SUMMARY_PROMPT,
    apply_retention,
    purge_session,
    summarise_session,
)
from tests.fakes.embedder import FakeEmbedder


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, messages, max_tokens=1200):
        self.calls.append(messages)
        return ("## Scenes\n- The cistern\n## NPCs met\n- The barkeep\n## Decisions\n- Went down\n"
                "## Loot\n- none\n## Open threads\n- Who owes gold?\n"
                "## Memorable lines\n- The smell of old coin.")


def _session(tmp: Path) -> Path:
    p = tmp / "sessions" / "20260911-1.jsonl"; p.parent.mkdir(parents=True)
    p.write_text("\n".join(json.dumps(r) for r in [
        {"ts": 1, "text": "The cistern is dark.", "decision": "IGNORE", "spoke": False},
        {"ts": 2, "text": "Understudy, smell?", "decision": "ADDRESSED", "spoke": True},
    ]) + "\n")
    return p


def test_prompt_forbids_player_names_and_ooc():
    assert "player" in SUMMARY_PROMPT.lower() and "out of character" in SUMMARY_PROMPT.lower()


def test_summarise_writes_file_and_indexes_both(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder()); llm = FakeLLM()
    p = _session(tmp_path)
    out = summarise_session(p, store=store, llm=llm)
    assert out == p.with_suffix(".summary.md") and "barkeep" in out.read_text()
    hits = store.search("barkeep", corpus="sessions", k=2)
    assert hits and hits[0].source.endswith(".summary.md")
    assert any(h.source.endswith(".jsonl") for h in store.search("cistern dark", corpus="sessions", k=3))


def test_summarise_twice_does_not_duplicate_chunks(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    p = _session(tmp_path)
    summarise_session(p, store=store, llm=FakeLLM())
    first = len(store.search("barkeep", corpus="sessions", k=50))
    summarise_session(p, store=store, llm=FakeLLM())
    assert len(store.search("barkeep", corpus="sessions", k=50)) == first


def test_retention_deletes_old_raw_keeps_summary(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    p = _session(tmp_path); summarise_session(p, store=store, llm=FakeLLM())
    old = time.time() - 10 * 86400
    import os; os.utime(p, (old, old))
    removed = apply_retention(p.parent, store=store, retention_days=7)
    assert removed == [p.name] and not p.exists() and p.with_suffix(".summary.md").exists()
    assert all(not h.source.endswith(".jsonl") for h in store.search("cistern", corpus="sessions", k=5))


def test_purge_removes_everything_for_session(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    p = _session(tmp_path); summarise_session(p, store=store, llm=FakeLLM())
    assert purge_session(p.parent, "20260911-1", store=store) is True
    assert not p.exists() and not p.with_suffix(".summary.md").exists()
    assert store.search("barkeep", corpus="sessions", k=3) == []


def test_purge_of_an_unknown_session_is_false_not_an_error(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    (tmp_path / "sessions").mkdir()
    assert purge_session(tmp_path / "sessions", "nope", store=store) is False


from familiar_brain.summariser import load_players, scrub_players  # noqa: E402


def test_scrub_players_drops_lines_and_counts():
    md = "## Scenes\n- The cistern\n- James rolled a natural twenty\n## NPCs met\n- Elias the barkeep\n"
    out, dropped = scrub_players(md, players=["James", "Elias"])
    assert dropped == 2 and "James" not in out and "Elias" not in out and "- The cistern" in out


def test_scrub_players_is_word_bounded_and_case_insensitive():
    out, dropped = scrub_players("- Jameson Street\n- jax laughed\n", players=["James", "Jax"])
    assert dropped == 1 and "Jameson Street" in out


def test_scrub_players_with_no_players_is_identity():
    md = "## Scenes\n- The cistern\n"
    assert scrub_players(md, players=[]) == (md, 0)


def test_load_players_skips_blanks_and_comments(tmp_path):
    p = tmp_path / "players.txt"
    p.write_text("# a comment\nJames\n\n  Elias  \n", encoding="utf-8")
    assert load_players(p) == ["James", "Elias"]


def test_load_players_missing_file_is_empty(tmp_path):
    assert load_players(tmp_path / "nope.txt") == []


def test_summarise_session_drops_lines_naming_players(tmp_path):
    class NamingLLM:
        def complete(self, messages, max_tokens=1200):
            return "## Scenes\n- The cistern\n- James rolled a twenty\n## NPCs met\n- The barkeep\n"

    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    p = _session(tmp_path)
    out = summarise_session(p, store=store, llm=NamingLLM(), players=["James"])
    assert "James" not in out.read_text() and "The cistern" in out.read_text()
    assert all("James" not in h.text for h in store.search("James twenty", corpus="sessions", k=10))
