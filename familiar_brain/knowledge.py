from __future__ import annotations

from familiar_brain.retrieval import Store

# Order matters: it is the order the model reads. The table's own rules come before
# the SRD so a house rule is the first thing seen on a rules question.
CORPORA = ("homebrew", "rules", "notes", "sessions")
CHUNK_CHARS = 700  # per-chunk cap so one long rule never crowds out the notes


def build_context(store: Store, query: str, *, chunks_per_corpus: int, max_chars: int) -> str:
    """The "What you know" block prepended to the system prompt on an ADDRESSED turn.

    Returns "" when nothing was retrieved, so the caller can prepend unconditionally.
    """
    lines: list[str] = []
    used = 0
    for corpus in CORPORA:
        for h in store.search(query, corpus=corpus, k=chunks_per_corpus):
            tag = f" ({h.meta['edition']} rules)" if corpus == "rules" and "edition" in h.meta else ""
            line = f"- [{corpus}] {h.title}{tag}: {h.text[:CHUNK_CHARS]}"
            if used + len(line) > max_chars:
                return _wrap(lines)
            lines.append(line)
            used += len(line)
    return _wrap(lines)


def _wrap(lines: list[str]) -> str:
    return "What you know:\n" + "\n".join(lines) if lines else ""
