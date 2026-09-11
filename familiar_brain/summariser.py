from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

from familiar_brain.chunking import markdown_chunks, split_soft
from familiar_brain.retrieval import Chunk, Store

log = logging.getLogger("familiar.summariser")

SUMMARY_PROMPT = (
    "You are the memory of a familiar at a Dungeons & Dragons table. Below is a transcript of what was said "
    "during one session, one line per utterance. Write a summary of the FICTION only, in Markdown with exactly "
    "these headings: ## Scenes, ## NPCs met, ## Decisions, ## Loot, ## Open threads, ## Memorable lines.\n"
    "Rules: never name or describe the players or anything said out of character; do not record dice results, "
    "table logistics, or jokes about the players; keep each bullet under 25 words; if a heading has nothing, "
    "write '- none'."
)


def load_players(path: Path | str) -> list[str]:
    """Real first names of the people at the table. Gitignored; absent is normal."""
    p = Path(path)
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def scrub_players(md: str, *, players: list[str]) -> tuple[str, int]:
    """Drop whole summary lines that name somebody at the table.

    SUMMARY_PROMPT already forbids it. This is the guard that does not depend on a
    small model obeying an instruction, because the cost of one leak is a familiar
    talking about real people at a table where it was only ever meant to talk about the
    fiction. Dropping the whole line rather than the name is deliberate: a bullet with
    the name cut out of it is still a sentence about a player.
    """
    if not players:
        return md, 0
    pat = re.compile(r"\b(" + "|".join(re.escape(p) for p in players) + r")\b", re.IGNORECASE)
    kept: list[str] = []
    dropped = 0
    for line in md.splitlines():
        if pat.search(line):
            dropped += 1
        else:
            kept.append(line)
    return "\n".join(kept) + ("\n" if md.endswith("\n") else ""), dropped


def summarise_session(path: Path, *, store: Store, llm, players: list[str] = ()) -> Path:
    """Write a fiction-only summary beside the raw log and index both.

    The summary carries weight 1.5 so it outranks the raw transcript on a tie: a
    written scene is a better answer than the line somebody happened to say.
    """
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    transcript = "\n".join(r["text"] for r in rows if r.get("text"))
    md = llm.complete([{"role": "system", "content": SUMMARY_PROMPT},
                       {"role": "user", "content": transcript}])
    md, dropped = scrub_players(md, players=list(players))
    if dropped:
        log.warning("summariser dropped %d line(s) naming players from %s", dropped, Path(path).name)
    out = Path(path).with_suffix(".summary.md")
    out.write_text(md, encoding="utf-8")
    sid = Path(path).stem
    # Delete first so re-summarising a session replaces its chunks instead of doubling them.
    store.delete_source(out.name)
    store.delete_source(Path(path).name)
    summary_chunks = [Chunk("sessions", out.name, f"Session {sid}: {c.title}", c.text, {"session": sid}, 1.5)
                      for c in markdown_chunks(md, source=out.name, corpus="sessions")]
    raw_chunks = [Chunk("sessions", Path(path).name, f"Session {sid} transcript", piece, {"session": sid}, 1.0)
                  for piece in split_soft(transcript)]
    store.add_chunks(summary_chunks + raw_chunks)
    log.info("summarised %s: %d summary chunks, %d transcript chunks",
             Path(path).name, len(summary_chunks), len(raw_chunks))
    return out


def summarise_in_background(path: Path, *, store: Store, llm,
                            players: list[str] = ()) -> threading.Thread:
    """Summarising is a several-second generation. The table should not wait on it."""
    def job() -> None:
        try:
            summarise_session(path, store=store, llm=llm, players=players)
        except Exception as e:
            log.error("summariser failed for %s: %s", Path(path).name, e)
    t = threading.Thread(target=job, daemon=True, name="summariser")
    t.start()
    return t


def apply_retention(sessions_dir: Path, *, store: Store, retention_days: int) -> list[str]:
    """Drop raw transcripts past the retention window. Summaries are kept indefinitely."""
    cutoff = time.time() - retention_days * 86400
    removed: list[str] = []
    for raw in sorted(Path(sessions_dir).glob("*.jsonl")):
        if raw.stat().st_mtime < cutoff:
            store.delete_source(raw.name)
            raw.unlink()
            removed.append(raw.name)
    if removed:
        log.info("retention removed %d raw transcript(s): %s", len(removed), ", ".join(removed))
    return removed


def purge_session(sessions_dir: Path, session_id: str, *, store: Store) -> bool:
    """Forget one session completely: raw log, summary, and every chunk of either."""
    raw = Path(sessions_dir) / f"{session_id}.jsonl"
    summ = raw.with_suffix(".summary.md")
    found = False
    for f in (raw, summ):
        if f.exists():
            store.delete_source(f.name)
            f.unlink()
            found = True
    log.info("purge %s: %s", session_id, "removed" if found else "nothing to remove")
    return found
