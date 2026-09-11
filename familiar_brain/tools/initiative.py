from __future__ import annotations


class Tracker:
    def __init__(self):
        self._entries: dict[str, int] = {}
        self._idx = -1

    def add(self, name: str, value: int) -> None:
        self._entries[name] = value

    def order(self) -> list[str]:
        return [n for n, _ in sorted(self._entries.items(), key=lambda kv: (-kv[1], kv[0]))]

    def show(self) -> list[tuple[str, int]]:
        return [(n, self._entries[n]) for n in self.order()]

    def next(self) -> str | None:
        order = self.order()
        if not order:
            return None
        self._idx = (self._idx + 1) % len(order)
        return order[self._idx]

    def clear(self) -> None:
        self._entries.clear(); self._idx = -1


TRACKER = Tracker()

SPEC = {
    "type": "function",
    "name": "initiative",
    "description": "Track combat turn order. Actions: add (name, value), next, show, clear.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "next", "show", "clear"]},
            "name": {"type": "string"},
            "value": {"type": "integer"},
        },
        "required": ["action"],
    },
}


def run(args: dict) -> str:
    a = args.get("action")
    if a == "add":
        TRACKER.add(str(args["name"]), int(args["value"]))
        return "Order: " + ", ".join(f"{n} ({v})" for n, v in TRACKER.show())
    if a == "next":
        n = TRACKER.next()
        return f"{n} is up." if n else "No one is in the order."
    if a == "show":
        rows = TRACKER.show()
        if not rows:
            return "Order is empty."
        return "Order: " + ", ".join(f"{n} ({v})" for n, v in rows)
    if a == "clear":
        TRACKER.clear(); return "Order cleared."
    return "Unknown initiative action."
