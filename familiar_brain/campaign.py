"""The campaign memory: one running record of what has happened in play, across sessions.

Each session summary answers questions about that session. Nothing answered "what are we
in the middle of?" across them: the open threads of three sessions ago sat in their own
summary, outranked by whatever the last session happened to say. So after each summary,
the model merges it into data/sessions/campaign-memory.md, which is indexed into the
"sessions" corpus above any single session.

It lives beside the summaries, not in data/notes: notes are written by people
(docs/notes-policy.md) and are the DM's canon; this is the familiar's own record of play,
and the prompt says so. The version before each merge is kept as campaign-memory.prev.md
so a bad merge can be undone by hand.

The first line records which summaries it was built from, each with a hash of its text.
That is what keeps it honest: a summary purged, re-summarised, or learnt out of order
(an older session caught up late) no longer matches, and the memory is rebuilt from every
remaining summary, oldest first, one merge at a time. Rebuilding rather than asking the
model to subtract one session is deliberate: a small model told to forget something is
not to be trusted to, and the summaries are the record to rebuild from.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from pathlib import Path
from typing import Callable

from familiar_brain.chunking import markdown_chunks
from familiar_brain.retrieval import Chunk, Store
from familiar_brain.summariser import EMPTY_SUMMARY, FICTION_RULES, ask, write_atomic

log = logging.getLogger("familiar.campaign")

CAMPAIGN_FILE = "campaign-memory.md"
PREV_FILE = "campaign-memory.prev.md"
# About 1500 tokens. Small enough that the merge (memory, one session summary, prompt and
# reply) stays under 7k tokens of a 16k slot, and that the whole of it is about ten
# chunks, of which build_context takes the few that match the question. Not more: the
# prompt budget is max_context_chars 4800 for every corpus together.
CAMPAIGN_CAP = 6000
# A memory at the cap is 2000 tokens at 3 characters a token, so 2000 left no room, and a
# reply cut off at max_tokens is now a failed merge (LLMClient.complete), not a memory
# short of its last headings.
CAMPAIGN_MAX_TOKENS = 2500
# Just above a session summary's 1.5, so on a tie the familiar's view of the whole campaign
# answers first, and below a deliberate note's 2.0. It was 2.0: measured with the BGE
# embedder on 2026-10-02, the campaign chunks then took all three sessions slots for "what
# was the seal on the letter?" and the session summary that named the seal never reached the
# prompt, though the memory had compressed the seal away.
CAMPAIGN_WEIGHT = 1.6
CAMPAIGN_HEADINGS = ("Story so far", "NPCs", "Places", "Open threads", "Loot and resources",
                     "Decisions that matter")
# Said in each chunk's title, which is embedded with its text, so the sections people ask
# about without naming them are found: "what should we do next?" says nothing of threads.
SECTION_CUES = {
    "Open threads": "what is still unresolved, and what the party could do next",
    "Story so far": "a recap of what has happened in the campaign so far",
}

CAMPAIGN_PROMPT = (
    "You keep the campaign memory of a familiar at a Dungeons & Dragons table: one running record of what "
    "has happened in play across every session. Below are the campaign memory so far and the summary of one "
    "more session. Write the whole updated campaign memory in Markdown with exactly these headings: "
    + ", ".join(f"## {h}" for h in CAMPAIGN_HEADINGS) + ".\n"
    f"Keep the whole memory under {CAMPAIGN_CAP:,} characters: shorten older story to make room for new. "
    "Carry forward every open thread that is still unresolved and drop the threads this session resolved. "
    "Record only what happened in play: the Dungeon Master's campaign notes are kept elsewhere and are canon, "
    "so you must not invent background or lore.\n"
    + FICTION_RULES
)

_HEADER = re.compile(r"^<!-- folded: (.*) -->\n")
# One merge or forget at a time, so a purge from a request thread cannot interleave with
# the learner writing the memory back.
_LOCK = threading.Lock()


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def foldable(sessions_dir: Path | str) -> list[tuple[str, str]]:
    """(session id, summary hash) for every summary with something in it, oldest first.
    Session ids start with the date and time, so their order is the order they were played."""
    out = []
    for p in sorted(Path(sessions_dir).glob("*.summary.md")):
        text = p.read_text(encoding="utf-8")
        if text != EMPTY_SUMMARY:
            out.append((p.name[: -len(".summary.md")], _hash(text)))
    return out


def folded(path: Path) -> list[tuple[str, str]]:
    """The summaries a campaign memory file was built from, in the order merged."""
    if not path.exists():
        return []
    m = _HEADER.match(path.read_text(encoding="utf-8"))
    return [tuple(pair.split("=", 1)) for pair in m.group(1).split()] if m else []


def read_body(path: Path) -> str:
    """The memory itself, without the bookkeeping line."""
    return _HEADER.sub("", path.read_text(encoding="utf-8"), count=1) if path.exists() else ""


def campaign_due(sessions_dir: Path | str) -> bool:
    """Whether the memory no longer matches the summaries: a cheap file read, so the learner
    can ask on every kick."""
    return folded(Path(sessions_dir) / CAMPAIGN_FILE) != foldable(sessions_dir)


_NONE = re.compile(r"^(?:[-*]\s*)?none\.?$", re.IGNORECASE)


def campaign_chunks(body: str) -> list[Chunk]:
    """One chunk per heading at least (markdown_chunks), so Open threads and Story so far are
    each retrievable on their own rather than diluted by the rest of the memory.

    A section that is only "- none" is left out: it answers nothing, and indexed it took a
    sessions slot ("Loot and resources: - none" ranked second for "what should we do next?")."""
    out = []
    for c in markdown_chunks(body, source=CAMPAIGN_FILE, corpus="sessions"):
        if all(_NONE.match(l.strip()) for l in c.text.splitlines() if l.strip()):
            continue
        cue = SECTION_CUES.get(c.title)
        title = f"Campaign memory: {c.title}" + (f" ({cue})" if cue else "")
        out.append(Chunk("sessions", CAMPAIGN_FILE, title, c.text, {"campaign": True}, CAMPAIGN_WEIGHT))
    return out


def _cap(md: str) -> str:
    """The prompt asks for the cap; this enforces it, should the model ignore it, by dropping
    the oldest lines of Story so far, the section that grows with every session, keeping its
    latest line and every other section.

    It used to cut from the end, which with Story so far first lost NPCs, Places, Open
    threads, Loot and Decisions, and the next merge built on what was left, so they were
    gone for good. A memory still over the cap with its story down to one line is refused:
    the merge fails, nothing is written, and the memory stays due."""
    if len(md) <= CAMPAIGN_CAP:
        return md
    log.warning("campaign memory came back at %d characters, over the %d cap; dropping its oldest story",
                len(md), CAMPAIGN_CAP)
    lines = md.splitlines(keepends=True)
    heads = [i for i, l in enumerate(lines) if l.startswith("#")]
    start = next((i for i in heads if _heading(lines[i]) == "story so far"), None)
    if start is not None:
        end = next((i for i in heads if i > start), len(lines))
        story = [i for i in range(start + 1, end) if lines[i].strip()]
        over = len(md) - CAMPAIGN_CAP
        drop: set[int] = set()
        for i in story[:-1]:
            if over <= 0:
                break
            drop.add(i)
            over -= len(lines[i])
        if over <= 0:
            return "".join(l for i, l in enumerate(lines) if i not in drop)
    raise ValueError(f"campaign memory of {len(md)} characters will not fit the {CAMPAIGN_CAP} cap "
                     "by shortening its story; merge refused")


def _heading(line: str) -> str:
    return line.lstrip("#").strip().lower()


def _fold(body: str, sid: str, summary: str, *, llm, players: list[str]) -> str:
    content = ("=== CAMPAIGN MEMORY SO FAR ===\n"
               + (body if body.strip() else "(nothing yet: this is the first session)\n")
               + f"\n=== SUMMARY OF SESSION {sid} ===\n{summary}")
    md = ask(llm, CAMPAIGN_PROMPT, content, players=players, what=f"campaign memory + {sid}",
             max_tokens=CAMPAIGN_MAX_TOKENS)
    return _cap(md if md.endswith("\n") else md + "\n")


def _write(d: Path, body: str, done: list[tuple[str, str]], store: Store) -> bool:
    with _LOCK:
        # A summary purged or rewritten while the model was merging: writing this would bring
        # back what it said. Left due, for the next refresh to rebuild.
        current = dict(foldable(d))
        if any(current.get(sid) != h for sid, h in done):
            log.warning("a session summary changed during the campaign merge; not written, will rebuild")
            return False
        mem = d / CAMPAIGN_FILE
        if mem.exists():
            os.replace(mem, d / PREV_FILE)
        header = "<!-- folded: " + " ".join(f"{sid}={h}" for sid, h in done) + " -->\n"
        write_atomic(mem, header + body)
        store.delete_source(CAMPAIGN_FILE)
        store.add_chunks(campaign_chunks(body))
    return True


def refresh_campaign(sessions_dir: Path | str, *, store: Store, llm, players: list[str] = (),
                     stop: Callable[[], bool] = lambda: False) -> int:
    """Bring the campaign memory up to date with the summaries. Returns how many merges ran.

    New summaries later than every one already merged are merged in one at a time, each
    written as it lands (so .prev is always one merge back). Anything else, a summary gone,
    changed, or older than one already merged, rebuilds from nothing, oldest first. `stop`
    is asked before each merge, so a session starting mid-way gets the GPU back; what was
    merged so far is a true memory of the sessions it names, and the rest follows later.
    A failed call raises with nothing written for that merge."""
    d = Path(sessions_dir)
    players = list(players)
    mem = d / CAMPAIGN_FILE
    want, have = foldable(d), folded(mem)
    if have == want:
        return 0
    if not want:
        with _LOCK:
            if mem.exists():
                mem.unlink()
            store.delete_source(CAMPAIGN_FILE)
        log.info("campaign memory removed: no session summaries remain")
        return 0
    if have and want[: len(have)] == have:
        body, done = read_body(mem), list(have)
    else:
        if have:
            log.info("campaign memory no longer matches the summaries; rebuilding from %d", len(want))
        body, done = "", []
    merged = 0
    for sid, h in want[len(done):]:
        if stop():
            log.info("campaign memory: a session started; %d merge(s) left for later", len(want) - len(done))
            break
        summary = (d / f"{sid}.summary.md").read_text(encoding="utf-8")
        body = _fold(body, sid, summary, llm=llm, players=players)
        done.append((sid, h))
        if not _write(d, body, done, store):
            break
        merged += 1
        log.info("campaign memory: merged %s (%d characters)", sid, len(body))
    return merged


def forget_session(sessions_dir: Path | str, sid: str, *, store: Store) -> bool:
    """Drop the campaign memory, and its previous version, if either holds `sid`. No model
    call: a purge must take effect at once, even mid-session. The learner rebuilds the
    memory from the summaries that remain at the next idle moment (campaign_due)."""
    d = Path(sessions_dir)
    dropped = False
    with _LOCK:
        for name in (CAMPAIGN_FILE, PREV_FILE):
            f = d / name
            if any(s == sid for s, _ in folded(f)):
                f.unlink()
                dropped = True
                if name == CAMPAIGN_FILE:
                    store.delete_source(CAMPAIGN_FILE)
    if dropped:
        log.info("campaign memory dropped: it held purged session %s; rebuilt when idle", sid)
    return dropped
