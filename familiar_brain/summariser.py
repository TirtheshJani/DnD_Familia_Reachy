from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

from familiar_brain.chunking import markdown_chunks, split_soft
from familiar_brain.retrieval import Chunk, Store
from familiar_brain.transcript import (
    DELIBERATE,
    DELIBERATE_WEIGHT,
    dedupe_revisions,
    piece_chunks,
    read_rows,
    transcript_pieces,
)

log = logging.getLogger("familiar.summariser")

# The fiction-only rules every prompt that writes memory shares: the session summary, its
# window partials and merge, and the campaign memory (campaign.py).
FICTION_RULES = (
    "Rules: never name or describe the players or anything said out of character; do not record dice results, "
    "table logistics, or jokes about the players; keep each bullet under 25 words; if a heading has nothing, "
    "write '- none'."
)
HEADINGS = "## Scenes, ## NPCs met, ## Decisions, ## Loot, ## Open threads, ## Memorable lines"
SUMMARY_PROMPT = (
    "You are the memory of a familiar at a Dungeons & Dragons table. Below is a transcript of what was said "
    "during one session, one line per utterance. Write a summary of the FICTION only, in Markdown with exactly "
    f"these headings: {HEADINGS}.\n" + FICTION_RULES
)
PART_PROMPT = (
    "You are the memory of a familiar at a Dungeons & Dragons table. Below is one part of the transcript of "
    "one session, one line per utterance; the parts before and after it are summarised separately. Write a "
    "summary of the FICTION in this part only, in Markdown with exactly these headings: "
    f"{HEADINGS}.\n" + FICTION_RULES
)
MERGE_PROMPT = (
    "You are the memory of a familiar at a Dungeons & Dragons table. Below are summaries of consecutive parts "
    "of one session, in order, and possibly the deliberate notes made during it. Merge them into one summary "
    f"of the FICTION, in Markdown with exactly these headings: {HEADINGS}. Say each thing once; a thread a "
    "later part resolved is no longer open; work every deliberate note in where it belongs.\n" + FICTION_RULES
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


NOTES_HEADING = "Deliberate notes"
EMPTY_SUMMARY = "Nothing was said in this session.\n"

# A summary's write and a purge of the same session, one at a time. The model calls take
# minutes and run outside it; only the check, the store swap and the file write are inside,
# so a purge waits at most for one embedding batch.
_LOCK = threading.Lock()


class Paused(Exception):
    """A session started before the summary's next model call. Nothing was written; the
    session is still unsummarised and is learnt at the next idle moment."""


class Purged(Exception):
    """The session was purged while it was being summarised. The result is dropped."""


# The most transcript one call is sent. llama-server runs -np 2 -c 32768, so a slot holds
# about 16k tokens. At the usual estimate of 4 characters a token this is 6k tokens; with
# the prompt (about 150) and the reply (max_tokens 1200) a call needs about 7.4k, and
# still under 9.5k if the transcript tokenises as badly as 3 characters a token (names,
# misheard words). That keeps every call well under 12k with room to spare in the slot.
# A four hour session is about 115k characters de-duplicated (the 21 minutes logged on
# 2026-09-16 were 10067), so about five windows; a session that fits takes one call.
WINDOW_CHARS = 24_000
# A merge folds up to a window of partials, each up to 1200 tokens, into one reply, and a
# reply cut off at max_tokens is now a failure (LLMClient.complete), so 1200 would fail a
# long session on every retry. 2000 keeps a merge under 10.2k tokens at 3 characters a token.
MERGE_MAX_TOKENS = 2000

# Whole lines made of nothing but these words carry no fiction: "Yeah.", "Uh.", "Okay okay."
# Only whole lines go, so "No, burn it." stays; deliberate notes are never dropped.
FILLER = frozenset("""
yeah yea yes yep yup no nope nah uh um umm uhm uh-huh mm mhm hmm hm oh ah eh ok okay right
sure so well like huh wow cool nice hey hi hello thanks alright
""".split())
_WORD = re.compile(r"[a-z0-9'-]+")


def is_filler(text: str) -> bool:
    return all(w in FILLER for w in _WORD.findall(text.lower()))


def _fit(text: str, cap: int) -> list[str]:
    """`text` in pieces of at most `cap` characters, cut at sentence ends where it can be.

    split_soft keeps a sentence whole even past its cap, and a run-on transcript line can
    be minutes of speech with no full stop at all, so a piece still over the cap is cut at
    the last space before it."""
    out: list[str] = []
    for part in split_soft(text, cap):
        while len(part) > cap:
            cut = part.rfind(" ", 0, cap + 1)
            cut = cut if cut > 0 else cap
            out.append(part[:cut].rstrip())
            part = part[cut:].lstrip()
        if part:
            out.append(part)
    return out


def windows(texts: list[str], cap: int = WINDOW_CHARS) -> list[str]:
    """Lines packed whole, in order, into windows of at most `cap` characters."""
    units = [u for t in texts for u in (_fit(t, cap) if len(t) > cap else [t])]
    out: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > cap:
            out.append(cur)
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur:
        out.append(cur)
    return out


def ask(llm, prompt: str, content: str, *, players: list[str], what: str, max_tokens: int = 1200,
        stop: Callable[[], bool] | None = None) -> str:
    """One memory-writing call, scrubbed. An empty reply is a failure like a timeout: an
    empty file would mark the session learnt and catch-up would never offer it again.

    `stop` is asked first and raises Paused if a session has started: a long session is
    several calls, each up to 1200 tokens on the GPU the table's replies wait for."""
    if stop is not None and stop():
        raise Paused(f"a session started; {what} left for later")
    md = llm.complete([{"role": "system", "content": prompt}, {"role": "user", "content": content}],
                      max_tokens=max_tokens)
    if not md or not md.strip():
        raise ValueError(f"the model returned nothing for {what}")
    md, dropped = scrub_players(md, players=players)
    if dropped:
        log.warning("summariser dropped %d line(s) naming players from %s", dropped, what)
    return md


def write_atomic(path: Path, text: str) -> None:
    """A crash mid-write must not leave a half file that reads as finished work."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _labelled(parts: list[str]) -> list[str]:
    return [f"=== Part {i} of {len(parts)} ===\n{p.strip()}" for i, p in enumerate(parts, 1)]


def _map_reduce(spoken: list[str], notes: list[str], *, llm, players: list[str], name: str,
                stop: Callable[[], bool] | None = None) -> str:
    """Summarise each window, then merge the partials into one summary.

    The partials are merged in stages while all of them together would not fit one window;
    each stage merges at least two, so it always ends. A partial or a merge is at most
    MERGE_MAX_TOKENS (about 8000 characters), so two always fit. The deliberate notes go, verbatim, only to
    the last merge, so the model can work them in where they belong."""
    wins = windows(spoken)
    parts = [ask(llm, PART_PROMPT, w, players=players, what=f"{name} part {i}/{len(wins)}", stop=stop)
             for i, w in enumerate(wins, 1)]
    notes_block = (f"## {NOTES_HEADING}\n" + "".join(f"- {n}\n" for n in notes)) if notes else ""
    while len(parts) > 1 and len("\n\n".join(_labelled(parts) + [notes_block])) > WINDOW_CHARS:
        groups: list[list[str]] = [[]]
        for p in parts:
            g = groups[-1]
            if len(g) >= 2 and len("\n\n".join(_labelled(g + [p]))) > WINDOW_CHARS:
                groups.append([p])
            else:
                g.append(p)
        parts = [ask(llm, MERGE_PROMPT, "\n\n".join(_labelled(g)), players=players, what=f"{name} merge",
                     max_tokens=MERGE_MAX_TOKENS, stop=stop)
                 if len(g) > 1 else g[0] for g in groups]
    content = "\n\n".join(_labelled(parts) + ([notes_block] if notes_block else []))
    return ask(llm, MERGE_PROMPT, content, players=players, what=f"{name} merge",
               max_tokens=MERGE_MAX_TOKENS, stop=stop)


def summarise_session(path: Path, *, store: Store, llm, players: list[str] = (),
                      stop: Callable[[], bool] | None = None) -> Path:
    """Write a fiction-only summary beside the raw log and index both.

    The summary carries weight 1.5 so it outranks the raw transcript on a tie: a
    written scene is a better answer than the line somebody happened to say.

    The transcript is de-duplicated first (transcript.dedupe_revisions): the live log holds
    each utterance up to six times as speech-to-speech re-sent it, which on 2026-09-16 was
    15607 characters for 9989 of distinct speech. The raw chunks are the same pieces the
    live indexer wrote under the same source, so this replaces them like for like.

    Deliberate notes are copied into the summary under their own heading, by code rather
    than by the model, so they outlive the raw log (apply_retention) whatever the model
    chose to keep, and are indexed at DELIBERATE_WEIGHT. They are scrubbed like the rest.

    A transcript longer than WINDOW_CHARS is summarised window by window and the partials
    merged (_map_reduce); filler lines are dropped first. Every call must succeed or
    nothing is written and the store is left as it was, live chunks included, so the
    session stays unsummarised and catch-up offers it again (learning.unsummarised).

    `stop` is asked before every model call; once it is true, Paused is raised with nothing
    written (ask). The result is written only if the raw log is still there, checked under
    the lock purge_session takes, else Purged is raised: a purge landing during the calls
    used to be undone by the write that followed them. The new chunks are embedded before
    the old ones go (Store.add_chunks replace), and the file is written last, so a failure at
    any step leaves the session unsummarised with its live chunks searchable.
    """
    path = Path(path)
    players = list(players)
    lines = dedupe_revisions(read_rows(path))
    out = path.with_suffix(".summary.md")
    sid = path.stem
    said = [r for r in lines if r.get("tag") == DELIBERATE or not is_filler(r["text"])]
    notes = [r["text"] for r in said if r.get("tag") == DELIBERATE]
    transcript = "\n".join(r["text"] for r in said)
    if not said:
        # No model call for a session where nothing was logged (one opened and closed at
        # once, or kept muted throughout), or nothing but "Yeah." and "Uh.". The file still
        # marks it learnt, so catch-up stops offering it and retention can let its raw log go.
        md = EMPTY_SUMMARY
    elif len(transcript) <= WINDOW_CHARS:
        md = ask(llm, SUMMARY_PROMPT, transcript, players=players, what=path.name, stop=stop)
    else:
        spoken = [r["text"] for r in said if r.get("tag") != DELIBERATE]
        md = _map_reduce(spoken, notes, llm=llm, players=players, name=path.name, stop=stop)
    if notes:
        md = md.rstrip("\n") + f"\n## {NOTES_HEADING}\n" + "".join(f"- {n}\n" for n in notes)
        md, dropped = scrub_players(md, players=players)
        if dropped:
            log.warning("summariser dropped %d note line(s) naming players from %s", dropped, path.name)
    summary_chunks = [] if md == EMPTY_SUMMARY else [
        Chunk("sessions", out.name, f"Session {sid}: {c.title}", c.text, {"session": sid},
              DELIBERATE_WEIGHT if c.title == NOTES_HEADING else 1.5)
        for c in markdown_chunks(md, source=out.name, corpus="sessions")]
    raw_chunks = piece_chunks(transcript_pieces(lines), sid=sid, source=path.name)
    with _LOCK:
        if not path.exists():
            raise Purged(f"{path.name} was purged while it was being summarised; result dropped")
        # The old chunks go in the same transaction the new ones arrive in, so re-summarising
        # replaces them instead of doubling them, and a failure keeps the live ones.
        store.add_chunks(summary_chunks + raw_chunks, replace=(out.name, path.name))
        write_atomic(out, md)
    if md == EMPTY_SUMMARY:
        log.info("summarised %s: nothing was said", path.name)
    log.info("summarised %s: %d summary chunks, %d transcript chunks",
             path.name, len(summary_chunks), len(raw_chunks))
    return out


def apply_retention(sessions_dir: Path, *, store: Store, retention_days: int,
                    keep: set[str] = frozenset()) -> list[str]:
    """Drop raw transcripts past the retention window. Summaries are kept indefinitely.

    Only a raw log that has been summarised is ever eligible: until then it is the only
    record of the session, and the one live session so far (2026-09-16) was never ended,
    so never summarised, and would have been deleted unlearnt a week later. `keep` names
    raw logs the learner still holds (learning.Learner.busy), so a summary landing while
    this runs cannot lose its log.
    """
    cutoff = time.time() - retention_days * 86400
    removed: list[str] = []
    for raw in sorted(Path(sessions_dir).glob("*.jsonl")):
        if raw.name in keep or not raw.with_suffix(".summary.md").exists():
            continue
        if raw.stat().st_mtime < cutoff:
            store.delete_source(raw.name)
            raw.unlink()
            removed.append(raw.name)
    if removed:
        log.info("retention removed %d raw transcript(s): %s", len(removed), ", ".join(removed))
    return removed


def purge_session(sessions_dir: Path, session_id: str, *, store: Store) -> bool:
    """Forget one session completely: raw log, summary, and every chunk of either.

    The campaign memory goes too if it holds the session (campaign.forget_session), with
    no model call, since a purge can come mid-session; the learner rebuilds it from the
    summaries that remain at the next idle moment."""
    # Imported here: campaign.py builds on this module's prompts and scrubbing.
    from familiar_brain.campaign import forget_session

    raw = Path(sessions_dir) / f"{session_id}.jsonl"
    summ = raw.with_suffix(".summary.md")
    found = False
    # Under the lock summarise_session writes under, so a summary of this session still in
    # its model calls finds the raw log gone and drops its result instead of writing it back.
    with _LOCK:
        for f in (raw, summ):
            # Whether or not the file is still there: chunks left under the name of a file
            # already gone would otherwise outlive every purge.
            found = store.delete_source(f.name) > 0 or found
            if f.exists():
                f.unlink()
                found = True
    forget_session(sessions_dir, session_id, store=store)
    log.info("purge %s: %s", session_id, "removed" if found else "nothing to remove")
    return found
