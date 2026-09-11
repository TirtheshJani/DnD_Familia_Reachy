from __future__ import annotations

from familiar_brain.retrieval import Store

_STORE: Store | None = None


def bind(store: Store) -> None:
    global _STORE
    _STORE = store


SPEC = {
    "type": "function",
    "name": "recall",
    "description": "Search the campaign notes and past session summaries for something the party has seen, met, decided, or been told.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "scope": {"type": "string", "enum": ["notes", "sessions", "all"]}},
        "required": ["query"],
    },
}


def run(args: dict) -> str:
    if _STORE is None:
        return "My memory is not open."
    scope = args.get("scope", "all")
    corpora = ["notes", "sessions"] if scope == "all" else [scope]
    lines = []
    for c in corpora:
        for h in _STORE.search(str(args.get("query", "")), corpus=c, k=3):
            lines.append(f"[{c}] {h.title}: {h.text[:400]}")
    return "\n".join(lines) if lines else "I recall nothing about that."
