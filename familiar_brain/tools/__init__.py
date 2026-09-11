"""Tool registry. Each tool module exposes SPEC (Responses API function tool schema) and run(args) -> str."""
from __future__ import annotations

from types import ModuleType

_REGISTRY: dict[str, ModuleType] = {}


def register(mod: ModuleType) -> None:
    _REGISTRY[mod.SPEC["name"]] = mod


def specs() -> list[dict]:
    return [m.SPEC for m in _REGISTRY.values()]


def run(name: str, args: dict) -> str:
    mod = _REGISTRY.get(name)
    if mod is None:
        return f"Unknown tool {name}."
    try:
        return mod.run(args)
    except Exception as e:  # the model narrates failures, the turn never crashes
        return f"The {name} tool failed: {type(e).__name__}: {e}"
