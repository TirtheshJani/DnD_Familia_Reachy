from __future__ import annotations

from familiar_brain.retrieval import Store

# Order matters: it is the order the model reads. The table's own rules come before
# the SRD so a house rule is the first thing seen on a rules question.
CORPORA = ("homebrew", "rules", "notes", "sessions")
CHUNK_CHARS = 700  # per-chunk cap so one long rule never crowds out the notes


# Said whenever the memory of play reaches the prompt. The campaign memory carries weight
# 2.0 within sessions, but weight only ranks inside a corpus: a misheard name recorded in
# play sat beside the DM's canon with nothing saying which was right. The merge prompt
# cannot settle it either, since the notes are never passed to the merge.
CANON_LINE = ("Where they disagree, [notes] are the Dungeon Master's canon and are right; "
              "[sessions] is only your memory of play.")


def build_context(store: Store, query: str, *, chunks_per_corpus: int, max_chars: int) -> str:
    """The "What you know" block prepended to the system prompt on an ADDRESSED turn.

    Each corpus's best hit is taken before any corpus's second, and a line that does not fit
    is skipped rather than ending the block. Taking corpora in turn, sessions (read last)
    was the first thing cut: three 700 character rules chunks left room for two sessions
    lines, and real campaign memory chunks run to 700 characters each. The lines are still
    written in CORPORA order.

    Returns "" when nothing was retrieved, so the caller can prepend unconditionally.
    """
    found: dict[str, list[str]] = {}
    for corpus in CORPORA:
        found[corpus] = []
        for h in store.search(query, corpus=corpus, k=chunks_per_corpus):
            tag = f" ({h.meta['edition']} rules)" if corpus == "rules" and "edition" in h.meta else ""
            found[corpus].append(f"- [{corpus}] {h.title}{tag}: {h.text[:CHUNK_CHARS]}")
    kept: dict[str, list[str]] = {c: [] for c in CORPORA}
    used = 0
    for rank in range(chunks_per_corpus):
        for corpus in CORPORA:
            if rank < len(found[corpus]) and used + len(line := found[corpus][rank]) <= max_chars:
                kept[corpus].append(line)
                used += len(line)
    return _wrap([line for c in CORPORA for line in kept[c]], canon=bool(kept["sessions"]))


def _wrap(lines: list[str], *, canon: bool = False) -> str:
    if not lines:
        return ""
    return "What you know:\n" + "\n".join(lines) + (f"\n{CANON_LINE}" if canon else "")
