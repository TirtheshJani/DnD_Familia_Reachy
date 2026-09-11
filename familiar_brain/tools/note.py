from __future__ import annotations

from familiar_brain.session import SessionLog

_LOG: SessionLog | None = None


def bind(log: SessionLog) -> None:
    global _LOG
    _LOG = log


SPEC = {
    "type": "function",
    "name": "note",
    "description": "Write a deliberate campaign note into the session log for later recall.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
}


def run(args: dict) -> str:
    if _LOG is None or _LOG.active_id is None:
        return "There is no session to note into."
    _LOG.append(text=str(args.get("text", "")), decision="ADDRESSED", spoke=False, tag="deliberate")
    return "Noted."
