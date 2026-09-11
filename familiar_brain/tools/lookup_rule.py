from __future__ import annotations

from familiar_brain.retrieval import Store

_STORE: Store | None = None


def bind(store: Store) -> None:
    global _STORE
    _STORE = store


SPEC = {
    "type": "function",
    "name": "lookup_rule",
    "description": "Look up a rule. Returns the table's house rule first if one exists, then the best passage from the 2014 and 2024 rules. Say which you are citing and defer to the Dungeon Master.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}


def run(args: dict) -> str:
    if _STORE is None:
        return "The rulebooks are not open."
    query = str(args.get("query", ""))
    lines = []
    house = _STORE.search(query, corpus="homebrew", k=2)
    for h in house:
        lines.append(f"House rule, {h.title}: {h.text[:500]}")
    best = {}
    for h in _STORE.search(query, corpus="rules", k=6):
        ed = h.meta.get("edition", "?")
        best.setdefault(ed, h)
    for ed, h in sorted(best.items()):
        lines.append(f"{ed} rules, {h.title}: {h.text[:600]}")
    if not lines:
        return "I found nothing on that in the rulebooks."
    if house:
        lines.append("The house rule takes precedence at this table; defer to the Dungeon Master.")
    return "\n".join(lines)
