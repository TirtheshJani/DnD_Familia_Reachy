"""TARS-style voice tool for reading and adjusting Rocky's humor, honesty, and behavior dials."""
from __future__ import annotations

import re
from typing import Callable

from familiar_brain.behavior import DIAL_NAMES, PRESETS, BehaviorSettings

_SETTINGS: BehaviorSettings | None = None
_ON_CHANGE: Callable[[dict], None] | None = None


def bind(settings: BehaviorSettings, on_change: Callable[[dict], None] | None = None) -> None:
    global _SETTINGS, _ON_CHANGE
    _SETTINGS = settings
    _ON_CHANGE = on_change


SPEC = {
    "type": "function",
    "name": "behavior_setting",
    "description": (
        "Read or adjust Rocky's TARS-style personality and creature behavior settings "
        "(humor, honesty, expressiveness, curiosity, bravery from 0 to 100, or preset). "
        "Use whenever a player asks for Rocky's humor/honesty setting or asks to dial one up or down."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["get", "set", "preset"]},
            "setting": {"type": "string", "enum": list(DIAL_NAMES)},
            "value": {"type": "integer", "minimum": 0, "maximum": 100},
            "preset": {"type": "string", "enum": list(PRESETS.keys())},
        },
        "required": ["action"],
    },
}

_ALIASES: dict[str, str] = {
    "humor": "humor",
    "humour": "humor",
    "funny": "humor",
    "honesty": "honesty",
    "honest": "honesty",
    "expressiveness": "expressiveness",
    "liveliness": "expressiveness",
    "curiosity": "curiosity",
    "music": "curiosity",
    "bravery": "bravery",
    "courage": "bravery",
    "grit": "bravery",
}

_SET_PAT = re.compile(
    r"\b(?:set|dial|turn|put|drop|raise|bump|change|adjust)\s+"
    r"(?:your\s+)?(?P<name>humou?r|honesty|expressiveness|liveliness|curiosity|bravery|courage|grit)"
    r"(?:\s+setting)?\s+(?:down\s+|up\s+|back\s+)?(?:to|at)\s+(?P<val>\d{1,3})\s*(?:%|percent)?\b",
    re.IGNORECASE,
)

_SHORT_SET_PAT = re.compile(
    r"\b(?P<name>humou?r|honesty|expressiveness|liveliness|curiosity|bravery)\s+"
    r"(?:to|at)\s+(?P<val>\d{1,3})\s*(?:%|percent)?\b",
    re.IGNORECASE,
)


def parse_voice_setting_command(text: str) -> tuple[str, int] | None:
    """Detect an explicit spoken dial command like 'Rocky, set humor to 60 percent'."""
    for pat in (_SET_PAT, _SHORT_SET_PAT):
        m = pat.search(text)
        if m:
            key = _ALIASES.get(m.group("name").lower())
            val = max(0, min(100, int(m.group("val"))))
            if key:
                return key, val
    return None


def run(args: dict) -> str:
    if _SETTINGS is None:
        return "Behavior settings are unavailable."
    action = str(args.get("action", "get")).lower()
    if action == "get":
        snap = _SETTINGS.snapshot()
        setting = args.get("setting")
        if setting in DIAL_NAMES:
            return f"{setting.capitalize()} is set to {getattr(snap, setting)} percent."
        return (
            f"Humor {snap.humor}%, Honesty {snap.honesty}%, "
            f"Expressiveness {snap.expressiveness}%, Curiosity {snap.curiosity}%, "
            f"Bravery {snap.bravery}%."
        )
    if action == "preset":
        preset = str(args.get("preset", "")).strip().lower()
        if preset not in PRESETS:
            return f"Unknown preset {preset!r}. Available: {', '.join(PRESETS)}."
        snap = _SETTINGS.update(preset=preset)
        if _ON_CHANGE is not None:
            _ON_CHANGE(snap.to_dict())
        return (
            f"Preset {preset} applied: Humor {snap.humor}%, Honesty {snap.honesty}%, "
            f"Expressiveness {snap.expressiveness}%, Curiosity {snap.curiosity}%, "
            f"Bravery {snap.bravery}%."
        )
    if action == "set":
        raw_name = str(args.get("setting", "")).strip().lower()
        key = _ALIASES.get(raw_name)
        if key not in DIAL_NAMES:
            return f"Unknown setting {raw_name!r}. Choose from {', '.join(DIAL_NAMES)}."
        if "value" not in args:
            return f"Specify a percentage from 0 to 100 for {key}."
        try:
            snap = _SETTINGS.update(**{key: args["value"]})
        except ValueError as exc:
            return str(exc)
        if _ON_CHANGE is not None:
            _ON_CHANGE(snap.to_dict())
        return f"{key.capitalize()} set to {getattr(snap, key)} percent."
    return "Unknown behavior_setting action."
