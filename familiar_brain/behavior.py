"""TARS-style personality, behavior, and privacy settings for Rocky.

All numeric dials run 0 to 100 and are persisted atomically to a local JSON file so
adjustments made from the Companion UI or by voice survive restarts. Dice roll totals
and rules citations are never altered by the honesty dial.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DIAL_NAMES: tuple[str, ...] = (
    "humor",
    "honesty",
    "expressiveness",
    "curiosity",
    "bravery",
)

DEFAULTS: dict[str, Any] = {
    "humor": 75,
    "honesty": 90,
    "expressiveness": 60,
    "curiosity": 50,
    "bravery": 55,
    "camera_enabled": False,
    "speaker_boost_db": 6.0,
}

PRESETS: dict[str, dict[str, Any]] = {
    "tars_default": {
        "humor": 75,
        "honesty": 90,
        "expressiveness": 60,
        "curiosity": 50,
        "bravery": 55,
    },
    "quiet_table": {
        "humor": 50,
        "honesty": 90,
        "expressiveness": 25,
        "curiosity": 20,
        "bravery": 50,
    },
    "playful_gremlin": {
        "humor": 95,
        "honesty": 85,
        "expressiveness": 85,
        "curiosity": 80,
        "bravery": 70,
    },
}


def _clamp_pct(value: Any, fallback: int) -> int:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return fallback
    return max(0, min(100, n))


def _clamp_boost_db(value: Any, fallback: float = 6.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    return round(max(0.0, min(15.0, n)), 1)


@dataclass(frozen=True)
class BehaviorSnapshot:
    humor: int = 75
    honesty: int = 90
    expressiveness: int = 60
    curiosity: int = 50
    bravery: int = 55
    camera_enabled: bool = False
    speaker_boost_db: float = 6.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "humor": self.humor,
            "honesty": self.honesty,
            "expressiveness": self.expressiveness,
            "curiosity": self.curiosity,
            "bravery": self.bravery,
            "camera_enabled": self.camera_enabled,
            "speaker_boost_db": self.speaker_boost_db,
        }


class BehaviorSettings:
    """Thread-safe, persisted dial bank for Rocky's voice tone, motion, and camera consent."""

    def __init__(self, path: Path | str | None = None, *, defaults: dict[str, Any] | None = None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._data: dict[str, Any] = dict(DEFAULTS)
        if defaults:
            for k in DIAL_NAMES:
                if k in defaults:
                    self._data[k] = _clamp_pct(defaults[k], DEFAULTS[k])
            if "camera_enabled" in defaults:
                self._data["camera_enabled"] = bool(defaults["camera_enabled"])
            if "speaker_boost_db" in defaults:
                self._data["speaker_boost_db"] = _clamp_boost_db(defaults["speaker_boost_db"])
        if self.path is not None:
            self.load()

    def load(self) -> BehaviorSnapshot:
        if self.path is None or not self.path.exists():
            return self.snapshot()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return self.snapshot()
        if not isinstance(raw, dict):
            return self.snapshot()
        with self._lock:
            for k in DIAL_NAMES:
                if k in raw:
                    self._data[k] = _clamp_pct(raw[k], DEFAULTS[k])
            if "camera_enabled" in raw:
                self._data["camera_enabled"] = bool(raw["camera_enabled"])
            if "speaker_boost_db" in raw:
                self._data["speaker_boost_db"] = _clamp_boost_db(raw["speaker_boost_db"])
            return BehaviorSnapshot(**self._data)

    def _save_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        tmp.write_text(json.dumps(self._data, indent=2) + "\n", encoding="utf-8")
        try:
            os.replace(tmp, self.path)
        except PermissionError:
            time.sleep(0.02)
            os.replace(tmp, self.path)

    def snapshot(self) -> BehaviorSnapshot:
        with self._lock:
            return BehaviorSnapshot(**self._data)

    def update(self, **kwargs: Any) -> BehaviorSnapshot:
        with self._lock:
            preset = kwargs.pop("preset", None)
            if preset is not None:
                key = str(preset).strip().lower()
                if key not in PRESETS:
                    raise ValueError(f"unknown preset: {preset!r}")
                for pk, pv in PRESETS[key].items():
                    self._data[pk] = pv
            for k, v in kwargs.items():
                if k in DIAL_NAMES:
                    if isinstance(v, bool):
                        raise ValueError(f"{k} must be an integer 0..100")
                    try:
                        num = float(v)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{k} must be a number 0..100") from exc
                    self._data[k] = max(0, min(100, int(round(num))))
                elif k == "camera_enabled":
                    if not isinstance(v, bool):
                        raise ValueError("camera_enabled must be a boolean")
                    self._data["camera_enabled"] = v
                elif k == "speaker_boost_db":
                    if isinstance(v, bool):
                        raise ValueError("speaker_boost_db must be a number 0..15")
                    try:
                        num = float(v)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("speaker_boost_db must be a number 0..15") from exc
                    self._data["speaker_boost_db"] = _clamp_boost_db(num)
                else:
                    raise ValueError(f"unknown setting: {k!r}")
            self._save_locked()
            return BehaviorSnapshot(**self._data)


def render_behavior_block(snap: BehaviorSnapshot) -> str:
    """Render the Humor and Honesty behavioral instructions for the system prompt."""
    h = snap.humor
    if h <= 15:
        humor_desc = "Keep your tone solemn, dry, and deadpan; avoid jokes or banter."
    elif h <= 45:
        humor_desc = "Use restrained, understated noir wit only when it fits naturally."
    elif h <= 80:
        humor_desc = (
            "Lean into crisp, dry TARS-style banter and playful teasing when addressed, "
            "staying concise and in character."
        )
    else:
        humor_desc = (
            "Be playfully mischievous and quick with cheeky familiar quips and comic timing, "
            "while still staying brief and respecting the Dungeon Master."
        )

    o = snap.honesty
    if o >= 95:
        honesty_desc = "Be completely blunt and unfiltered in your opinions with zero sugarcoating."
    elif o >= 75:
        honesty_desc = (
            "Be candid and straightforward in your opinions, tempering bluntness just enough "
            "for diplomacy at the table."
        )
    elif o >= 40:
        honesty_desc = "Be tactful, diplomatic, and slightly guarded when giving opinions or advice."
    else:
        honesty_desc = (
            "Answer subjective opinions with sly understatement, playful misdirection, or "
            "cryptic familiar riddles."
        )

    return (
        f"Current behavioral settings (adjustable on request via behavior_setting): "
        f"Humor {h}%, Honesty {o}%, Expressiveness {snap.expressiveness}%, "
        f"Curiosity {snap.curiosity}%, Bravery {snap.bravery}%.\n"
        f"Humor guidance: {humor_desc}\n"
        f"Honesty guidance: {honesty_desc} "
        "Hard invariant: never falsify dice roll totals, rules citations, or campaign facts "
        "regardless of the honesty setting."
    )


def scale_reaction_gaps(
    min_gap_s: float, reaction_gap_s: float, expressiveness: int
) -> tuple[float, float] | None:
    """Scale gesture and reaction cooldowns from the expressiveness dial (0..100).

    Returns None when expressiveness is 0 (non-verbal reactions paused).
    """
    exp = max(0, min(100, int(expressiveness)))
    if exp == 0:
        return None
    # 60% = 1.0x baseline; 100% = ~0.6x faster; 10% = ~2.6x slower (quiet motor footprint).
    factor = max(0.55, min(3.0, 1.6 - (exp / 100.0)))
    return (round(min_gap_s * factor, 2), round(reaction_gap_s * factor, 2))


def tilt_threat_mood(mood: str | None, bravery: int) -> str | None:
    """Tilt scared/angry creature reactions using the bravery dial (0..100)."""
    if mood is None:
        return None
    b = max(0, min(100, int(bravery)))
    if mood == "scared" and b >= 75:
        return "angry"
    if mood == "angry" and b <= 25:
        return "scared"
    return mood
