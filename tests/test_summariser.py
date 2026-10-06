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


# Long sessions: map-reduce over windows (deviation P18). A four hour session is about
# 115k characters after de-duplication and llama-server gives each slot about 16k tokens.

import math  # noqa: E402

import pytest  # noqa: E402

from familiar_brain.summariser import (  # noqa: E402
    MERGE_PROMPT,
    NOTES_HEADING,
    PART_PROMPT,
    WINDOW_CHARS,
)

PART = ("## Scenes\n- A part\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n"
        "## Open threads\n- none\n## Memorable lines\n- none\n")


class RecordingLLM:
    """Records every call; replies with `reply(messages)`, or raises on call number `fail_on`."""

    def __init__(self, reply=lambda m: PART, fail_on=None):
        self.calls = []
        self.reply = reply
        self.fail_on = fail_on

    def complete(self, messages, max_tokens=1200):
        self.calls.append(messages)
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("upstream timed out")
        return self.reply(messages)

    def by_prompt(self, prompt):
        return [m for m in self.calls if m[0]["content"] == prompt]


def _line(i: int, words: int = 30) -> str:
    # Each line its own vocabulary, so no line reads as a revision of the one before.
    return " ".join(f"w{i}x{j}" for j in range(words)) + "."


def _write_rows(tmp: Path, rows: list[dict], sid: str = "20260916-1") -> Path:
    p = tmp / "sessions" / f"{sid}.jsonl"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _long_session(tmp: Path, target_chars: int, notes: list[str] = ()) -> tuple[Path, list[str]]:
    lines, total, i = [], 0, 0
    while total < target_chars:
        lines.append(_line(i)); total += len(lines[-1]) + 1; i += 1
    rows = [{"ts": n, "text": t, "decision": "IGNORE", "spoke": False} for n, t in enumerate(lines)]
    for k, note in enumerate(notes):
        rows.insert(len(rows) // 2 + k, {"ts": 0, "text": note, "decision": "ADDRESSED", "spoke": False,
                                          "tag": "deliberate"})
    return _write_rows(tmp, rows), lines


def test_a_short_session_takes_one_call(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder()); llm = RecordingLLM()
    summarise_session(_session(tmp_path), store=store, llm=llm)
    assert len(llm.calls) == 1 and llm.calls[0][0]["content"] == SUMMARY_PROMPT


def test_a_long_session_is_summarised_window_by_window_then_merged(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder()); llm = RecordingLLM()
    p, lines = _long_session(tmp_path, int(WINDOW_CHARS * 3.5))
    summarise_session(p, store=store, llm=llm)
    parts = llm.by_prompt(PART_PROMPT)
    total = sum(len(l) + 1 for l in lines)
    assert math.ceil(total / WINDOW_CHARS) <= len(parts) <= math.ceil(total / WINDOW_CHARS) + 1
    assert all(len(m[1]["content"]) <= WINDOW_CHARS for m in parts)
    # In order, and nothing said is left out of every window.
    assert "\n".join(m[1]["content"] for m in parts).split("\n") == lines
    assert llm.calls[-1][0]["content"] == MERGE_PROMPT and len(llm.calls) == len(parts) + 1
    assert p.with_suffix(".summary.md").read_text() == PART


def test_a_run_on_line_longer_than_a_window_is_cut_to_fit(tmp_path):
    # A run-on with no sentence end at all: split_soft alone would keep it whole.
    words = [f"r{j}" for j in range(WINDOW_CHARS // 3)]
    p = _write_rows(tmp_path, [{"ts": 1, "text": " ".join(words), "decision": "IGNORE", "spoke": False}])
    llm = RecordingLLM()
    summarise_session(p, store=Store(tmp_path / "k.db", embedder=FakeEmbedder()), llm=llm)
    parts = llm.by_prompt(PART_PROMPT)
    assert len(parts) >= 2 and all(len(m[1]["content"]) <= WINDOW_CHARS for m in parts)
    assert " ".join(m[1]["content"] for m in parts).split() == words


def test_no_call_is_sent_more_than_a_window_even_when_the_parts_are_long(tmp_path):
    # Partials near the 1200 token output limit: merging them all at once would not fit.
    llm = RecordingLLM(reply=lambda m: "## Scenes\n" + "- a long scene line here.\n" * 200)
    p, _ = _long_session(tmp_path, WINDOW_CHARS * 8)
    summarise_session(p, store=Store(tmp_path / "k.db", embedder=FakeEmbedder()), llm=llm)
    assert len(llm.by_prompt(MERGE_PROMPT)) >= 2
    assert all(len(m[1]["content"]) <= WINDOW_CHARS for m in llm.calls)


def test_filler_and_revisions_are_dropped_before_summarising(tmp_path):
    rows = [{"ts": 1, "text": "Yeah.", "decision": "IGNORE", "spoke": False},
            {"ts": 2, "text": "", "decision": "IGNORE", "spoke": False},
            {"ts": 3, "text": "Uh, um.", "decision": "IGNORE", "spoke": False},
            {"ts": 4, "text": "The gate of the old keep is barred", "decision": "IGNORE", "spoke": False},
            {"ts": 5, "text": "The gate of the old keep is barred from inside.", "decision": "IGNORE",
             "spoke": False},
            {"ts": 6, "text": "No.", "decision": "IGNORE", "spoke": False},
            {"ts": 7, "text": "Burn it.", "decision": "IGNORE", "spoke": False}]
    llm = RecordingLLM()
    summarise_session(_write_rows(tmp_path, rows), store=Store(tmp_path / "k.db", embedder=FakeEmbedder()),
                      llm=llm)
    assert llm.calls[0][1]["content"] == "The gate of the old keep is barred from inside.\nBurn it."


def test_a_session_of_nothing_but_filler_takes_no_call(tmp_path):
    rows = [{"ts": 1, "text": "Yeah.", "decision": "IGNORE", "spoke": False},
            {"ts": 2, "text": "Okay okay.", "decision": "IGNORE", "spoke": False}]
    llm = RecordingLLM()
    out = summarise_session(_write_rows(tmp_path, rows), store=Store(tmp_path / "k.db", embedder=FakeEmbedder()),
                            llm=llm)
    assert llm.calls == [] and out.exists()


@pytest.mark.parametrize("fail_on", [1, 2, 4])
def test_any_failed_call_leaves_no_summary_and_keeps_the_live_chunks(tmp_path, fail_on):
    from familiar_brain.transcript import dedupe_revisions, piece_chunks, read_rows, transcript_pieces
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    p, _ = _long_session(tmp_path, WINDOW_CHARS * 3)
    live = piece_chunks(transcript_pieces(dedupe_revisions(read_rows(p))), sid=p.stem, source=p.name)
    store.add_chunks(live)
    llm = RecordingLLM(fail_on=fail_on)
    with pytest.raises(RuntimeError):
        summarise_session(p, store=store, llm=llm)
    assert len(llm.calls) == fail_on  # stops at the failure, sends nothing after it
    assert sorted(f.name for f in p.parent.iterdir()) == [p.name]  # no summary, no temp file
    n = store.db.execute("SELECT COUNT(*) FROM chunks WHERE source = ?", (p.name,)).fetchone()[0]
    assert n == len(live)


def test_an_empty_reply_counts_as_a_failure(tmp_path):
    p = _session(tmp_path)
    with pytest.raises(ValueError):
        summarise_session(p, store=Store(tmp_path / "k.db", embedder=FakeEmbedder()),
                          llm=RecordingLLM(reply=lambda m: "  \n"))
    assert not p.with_suffix(".summary.md").exists()


def test_deliberate_notes_reach_the_merge_prompt_verbatim(tmp_path):
    note = "Bram owes us ten gold, collect at the Gilded Eel."
    llm = RecordingLLM()
    p, _ = _long_session(tmp_path, WINDOW_CHARS * 2, notes=[note])
    summarise_session(p, store=Store(tmp_path / "k.db", embedder=FakeEmbedder()), llm=llm)
    merge = llm.by_prompt(MERGE_PROMPT)[-1][1]["content"]
    assert f"## {NOTES_HEADING}\n- {note}" in merge
    assert all(note not in m[1]["content"] for m in llm.by_prompt(PART_PROMPT))
    assert f"- {note}" in p.with_suffix(".summary.md").read_text()


def test_partials_and_the_merged_summary_are_scrubbed(tmp_path):
    llm = RecordingLLM(reply=lambda m: "## Scenes\n- The crypt\n- James rolled a twenty\n")
    p, _ = _long_session(tmp_path, WINDOW_CHARS * 2)
    out = summarise_session(p, store=Store(tmp_path / "k.db", embedder=FakeEmbedder()), llm=llm,
                            players=["James"])
    assert "James" not in llm.by_prompt(MERGE_PROMPT)[-1][1]["content"]
    assert "James" not in out.read_text() and "The crypt" in out.read_text()


def test_a_purge_forgets_chunks_whose_file_is_already_gone(tmp_path):
    """Purge deleted the chunks only of files that still existed, so chunks left under a
    raw log's name after the log itself was gone (a summary racing a purge, or a crash)
    survived every purge and Rocky could still quote the session."""
    from familiar_brain.retrieval import Chunk
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    d = tmp_path / "sessions"; d.mkdir()
    store.add_chunks([Chunk("sessions", "20261002-aaa.jsonl", "Session 20261002-aaa transcript",
                            "Bram hides the key.", {"session": "20261002-aaa"}),
                      Chunk("sessions", "20261002-aaa.summary.md", "Session 20261002-aaa: Scenes",
                            "- Bram hides the key", {"session": "20261002-aaa"}, 1.5)])
    purge_session(d, "20261002-aaa", store=store)
    assert store.search("key", corpus="sessions", k=5) == []


def test_an_embedder_failure_leaves_the_live_chunks_and_no_summary(tmp_path):
    """The old chunks were deleted and the summary file written before the new chunks were
    embedded. An embedder failure then left a summary file, which marks the session learnt,
    and nothing at all in the store: never searchable again and never retried."""
    from familiar_brain.transcript import dedupe_revisions, piece_chunks, read_rows, transcript_pieces

    class FlakyEmbedder(FakeEmbedder):
        down = False

        def embed(self, texts):
            if self.down:
                raise RuntimeError("embedder down")
            return super().embed(texts)

    emb = FlakyEmbedder()
    store = Store(tmp_path / "k.db", embedder=emb)
    p = _session(tmp_path)
    live = piece_chunks(transcript_pieces(dedupe_revisions(read_rows(p))), sid=p.stem, source=p.name)
    store.add_chunks(live)
    emb.down = True
    with pytest.raises(RuntimeError, match="embedder down"):
        summarise_session(p, store=store, llm=FakeLLM())
    assert sorted(f.name for f in p.parent.iterdir()) == [p.name]
    assert store.db.execute("SELECT DISTINCT source FROM chunks").fetchall() == [(p.name,)]
