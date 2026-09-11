"""Tool registry. Each tool module exposes SPEC (Responses API function tool schema) and run(args) -> str."""
from __future__ import annotations

import json
import logging
from types import ModuleType

log = logging.getLogger("familiar.tools")

_REGISTRY: dict[str, ModuleType] = {}


def register(mod: ModuleType) -> None:
    _REGISTRY[mod.SPEC["name"]] = mod


def specs() -> list[dict]:
    return [m.SPEC for m in _REGISTRY.values()]


def run(name: str, args: dict) -> str:
    mod = _REGISTRY.get(name)
    if mod is None:
        log.warning("unknown tool %s", name)
        return f"Unknown tool {name}."
    try:
        result = mod.run(args)
    except Exception as e:  # the model narrates failures, the turn never crashes
        log.warning("tool %s failed: %s: %s", name, type(e).__name__, e)
        return f"The {name} tool failed: {type(e).__name__}: {e}"
    # The audit trail for the table: what was asked, and what the tool actually returned.
    log.info("tool %s args=%s -> %s", name, json.dumps(args, ensure_ascii=False, sort_keys=True), result)
    return result
