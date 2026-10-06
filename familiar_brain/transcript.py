"""Reading a session log as a transcript: the one view of it that both the live indexer
(live_index.py) and the end-of-session summariser (summariser.py) index, so the chunks
the summariser writes replace the live ones like for like."""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from familiar_brain.chunking import SOFT_CAP, split_soft
from familiar_brain.retrieval import Chunk

DELIBERATE = "deliberate"
# Above the summary's 1.5: somebody asked for this to be remembered, which is a better
# answer than anything the familiar happened to overhear or the model chose to write up.
DELIBERATE_WEIGHT = 2.0
PIECE_CAP = SOFT_CAP

# How much of the earlier line the later one must repeat, in order, to be its revision.
# The live revisions on 2026-09-16 repeat 80 to 100 percent of the words they replace,
# misheard words aside ("pick up" became "piss up", "attainments" became "appealments");
# two neighbouring lines that are not revisions share a word or two at most.
REVISION_COVERAGE = 0.6
# Below this many words, 60 percent is two or three words, which any question about the
# line repeats; such a line needs a stricter test (is_revision).
SHORT_LINE_WORDS = 6

_WORD = re.compile(r"[a-z0-9']+")


def read_rows(path: Path | str) -> list[dict]:
    """Every whole row of a session log. A line that does not parse is skipped rather than
    fatal: the log is appended to while it is read, so the last line can be half written."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def is_revision(earlier: str, later: str) -> bool:
    """Whether `later` is speech-to-speech re-sending `earlier` with more audio behind it.

    With smart_turn on, it asks for a reply at the player's first pause and, if the player
    carries on, sends a fresh transcript of the longer audio (app._rewind_if_revision). The
    brain logs both, so the live log holds the same utterance 2 to 6 times, each a slightly
    different hearing of the last. Matched on words in order, not on a shared prefix: the
    re-hearing sometimes gains words at the front too ("I put out the man. Are y'all
    selecting ..." after "Are y'all selecting ...", 65 s apart), which also rules out a
    time window.

    A line of fewer than SHORT_LINE_WORDS words is a revision only if the later line begins
    with every word of it, give or take one word gained at the front. Coverage alone let a
    question fold away the fact it asked about: "The innkeeper is called Bram." then "Who is
    the innkeeper called again?" kept only the question, and "Goblins!" vanished into "How
    many goblins are there?". The short re-hearings on 2026-09-16 all begin with the earlier
    line ("Um the general vibe." grew at the end, "So excited." became "I'm so excited. ...");
    one that does not is kept twice, which costs a few words rather than a fact.
    """
    a, b = _WORD.findall(earlier.lower()), _WORD.findall(later.lower())
    if not a:
        return True
    blocks = [m for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks() if m.size]
    matched = sum(m.size for m in blocks)
    if len(a) < SHORT_LINE_WORDS:
        return matched == len(a) and blocks[0].a == 0 and blocks[0].b <= 1
    return matched / len(a) >= REVISION_COVERAGE


def _lost_lead(earlier: str, later: str) -> str:
    """The front of `earlier` that its revision `later` no longer has, or "".

    On a long run-on the transcriber's window can slide past the start: at 18:44:54 on
    2026-09-16 the revision began "It's not everything." and dropped the sentence before
    it. Replacing the line outright would lose words that were said; they are kept in
    front. Three words or more, so a misheard first word is not repeated.
    """
    a, b = _WORD.findall(earlier.lower()), _WORD.findall(later.lower())
    blocks = [m for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks() if m.size]
    if not blocks or blocks[0].b != 0 or blocks[0].a < 3:
        return ""
    cut = list(_WORD.finditer(earlier.lower()))[blocks[0].a].start()
    return earlier[:cut].rstrip()


def fold_row(lines: list[dict], row: dict) -> None:
    """Add one log row to an already de-duplicated list, in place.

    A revision replaces the line it revises, where that line stood. Only the latest spoken
    line can be revised, and deliberate notes are never folded either way: they are the
    note tool's words, not a hearing of the table, and a note made mid-utterance must not
    keep the revision after it from replacing the line before it.
    """
    if not row.get("text"):
        return
    if row.get("tag") != DELIBERATE:
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].get("tag") == DELIBERATE:
                continue
            if is_revision(lines[i]["text"], row["text"]):
                lead = _lost_lead(lines[i]["text"], row["text"])
                lines[i] = {**row, "text": f"{lead} {row['text']}"} if lead else row
                return
            break
    lines.append(row)


def dedupe_revisions(rows: list[dict]) -> list[dict]:
    lines: list[dict] = []
    for r in rows:
        fold_row(lines, r)
    return lines


def transcript_pieces(lines: list[dict], cap: int = PIECE_CAP) -> list[str]:
    """The spoken lines packed whole into pieces of up to `cap` characters, in order.

    Packed by line, not split_soft over the joined text, so a piece once made stays the same
    as the log grows: only the last piece changes when a line is added or revised. That is
    what lets the live indexer embed just the tail on each batch. Deliberate notes are left
    out; they are indexed on their own, at their own weight.
    """
    units: list[str] = []
    for line in lines:
        if line.get("tag") == DELIBERATE:
            continue
        text = line["text"].strip()
        units += split_soft(text, cap) if len(text) > cap else [text]
    pieces: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > cap:
            pieces.append(cur)
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur:
        pieces.append(cur)
    return pieces


def piece_chunks(pieces: list[str], *, sid: str, source: str, first: int = 0) -> list[Chunk]:
    """Transcript chunks, each stamped with its piece number so the live indexer can replace
    a changed tail without touching the pieces before it (Store.delete_pieces)."""
    return [Chunk("sessions", source, f"Session {sid} transcript", p, {"session": sid, "piece": first + i}, 1.0)
            for i, p in enumerate(pieces[first:])]


def note_chunks(notes: list[dict], *, sid: str, source: str) -> list[Chunk]:
    return [Chunk("sessions", source, f"Session {sid} note", n["text"], {"session": sid, "deliberate": True},
                  DELIBERATE_WEIGHT) for n in notes]
