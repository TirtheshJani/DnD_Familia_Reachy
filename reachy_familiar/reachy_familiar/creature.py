"""Living-pet creature motion engine for Reachy Mini (firmware 1.15.0+).

Maps Rocky's moods to verified moves from pollen-robotics/reachy-mini-emotions-library,
drives speech-synced wobbling, true antennas-down Deep Night / Mute sleep poses, DoA
sound-direction glances, and subtle background-music sway while enforcing quiet motor
rate limits so Rocky stays charming without distracting the table.
"""
from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import Any

import requests

log = logging.getLogger("reachy_familiar.creature")

DATASET = "pollen-robotics/reachy-mini-emotions-library"

# Verified against GET /api/move/recorded-move-datasets/list/pollen-robotics/reachy-mini-emotions-library
# on Reachy Mini Wireless firmware 1.15.0.
EMOTION_LIBRARY: dict[str, tuple[str, ...]] = {
    "happy": ("cheerful1", "laughing1"),
    "amused": ("laughing1", "laughing2", "cheerful1"),
    "funny": ("laughing2", "laughing1", "oops1"),
    "joyful": ("enthusiastic1", "enthusiastic2", "amazed1"),
    "angry": ("furious1", "irritated1", "irritated2", "rage1"),
    "scared": ("scared1", "fear1", "anxiety1", "shy1"),
    "sad": ("sad1", "sad2", "downcast1", "lonely1"),
    "surprised": ("surprised1", "surprised2", "electric1"),
    "alarmed": ("surprised1", "surprised2"),
    "curious": ("curious1", "inquiring1", "inquiring2"),
    "thinking": ("thoughtful1", "thoughtful2"),
    "thoughtful": ("thoughtful1", "thoughtful2"),
    "attentive": ("attentive1", "attentive2"),
    "success": ("success1", "success2", "proud1"),
    "proud": ("proud1", "proud2", "proud3"),
    "cheer": ("success1", "enthusiastic1", "cheerful1"),
    "sleepy": ("mini-deep-sleep", "sleep1"),
    "asleep": ("mini-deep-sleep",),
    "welcoming": ("welcoming1", "welcoming2", "wake-mini-up"),
    "awake": ("welcoming1", "wake-mini-up"),
    "calm": ("serenity1", "calming1"),
}

ALL_VALID_MOVES: frozenset[str] = frozenset(
    move for group in EMOTION_LIBRARY.values() for move in group
)


def resolve_emotion_move(intent: str, *, rng: random.Random | None = None) -> str:
    """Resolve an intent name (e.g. 'angry', 'joyful') or raw move id to a verified Reachy move."""
    clean = str(intent or "").strip().lower()
    if clean in ALL_VALID_MOVES:
        return clean
    choices = EMOTION_LIBRARY.get(clean)
    if choices:
        picker = rng or random
        return picker.choice(choices)
    return "attentive1"


class CreatureController:
    """Executes emotions, face tracking, speech wobbling, sleep poses, and ambient sway."""

    def __init__(
        self,
        *,
        daemon_url: str = "http://127.0.0.1:8000",
        reachy_mini: Any = None,
        session: requests.Session | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.daemon_url = daemon_url.rstrip("/")
        self.reachy = reachy_mini
        self._http = session or requests.Session()
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._last_emotion_at = 0.0
        self._last_ambient_at = 0.0
        self._last_sway_at = 0.0
        self._sway_phase = 0.0
        self._sleeping_pose = False
        self._wobbling = False
        self._tracking = False

    def _post(self, path: str, payload: dict[str, Any] | None = None, timeout: float = 2.0) -> bool:
        try:
            r = self._http.post(f"{self.daemon_url}{path}", json=payload, timeout=timeout)
            return r.ok
        except Exception as exc:
            log.debug("daemon POST %s failed: %s", path, exc)
            return False

    def _get(self, path: str, timeout: float = 1.0) -> dict[str, Any] | None:
        try:
            r = self._http.get(f"{self.daemon_url}{path}", timeout=timeout)
            if r.ok:
                return r.json()
        except Exception:
            pass
        return None

    def set_tracking(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        with self._lock:
            if enabled == self._tracking:
                return True
            self._tracking = enabled
        path = "/api/media/tracking/enable" if enabled else "/api/media/tracking/disable"
        return self._post(path)

    def set_wobbling(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        with self._lock:
            if enabled == self._wobbling:
                return True
            self._wobbling = enabled
        path = "/api/media/wobbling/enable" if enabled else "/api/media/wobbling/disable"
        return self._post(path)

    def enter_sleep_pose(self) -> bool:
        with self._lock:
            if self._sleeping_pose:
                return True
            self._sleeping_pose = True
        self.set_wobbling(False)
        self.set_tracking(False)
        return self._post("/api/move/play/goto_sleep")

    def wake_from_sleep_pose(self) -> bool:
        with self._lock:
            was_sleeping = self._sleeping_pose
            self._sleeping_pose = False
        if not was_sleeping:
            return True
        ok = self._post("/api/move/play/wake_up")
        self.set_tracking(True)
        return ok

    def play_emotion(self, intent: str, *, force: bool = False) -> str | None:
        """Play a verified recorded emotion move on Reachy Mini."""
        clean = str(intent or "").strip().lower()
        if clean in ("sleepy", "asleep", "mini-deep-sleep"):
            self.enter_sleep_pose()
            return "mini-deep-sleep"
        if clean in ("welcoming", "awake", "wake-mini-up") and self._sleeping_pose:
            self.wake_from_sleep_pose()
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_emotion_at) < 3.5:
                return None
            self._last_emotion_at = now
        move = resolve_emotion_move(clean, rng=self._rng)
        self._post(f"/api/move/play/recorded-move-dataset/{DATASET}/{move}")
        return move

    def stop_emotion(self) -> bool:
        return self._post("/api/move/stop")

    def move_head(self, direction: str) -> bool:
        """Gentle, quiet clamped head glance in a cardinal direction."""
        angles = {
            "left": (0.0, 0.0, 0.35),
            "right": (0.0, 0.0, -0.35),
            "up": (0.0, -0.22, 0.0),
            "down": (0.0, 0.25, 0.0),
            "front": (0.0, 0.0, 0.0),
        }
        roll, pitch, yaw = angles.get(str(direction).strip().lower(), (0.0, 0.0, 0.0))
        return self._post(
            "/api/move/goto",
            {
                "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": roll, "pitch": pitch, "yaw": yaw},
                "duration": 0.65,
            },
        )

    def tick_ambient(
        self,
        *,
        ambient: dict[str, Any],
        behavior: dict[str, Any],
        muted: bool,
        asleep: bool,
        speaking: bool,
        now: float | None = None,
    ) -> str | None:
        """Drive quiet, rate-limited living-pet posture & music reactions between spoken turns.

        Returns a short tag describing any action taken (for logs/tests), or None if idle.
        """
        ts = time.monotonic() if now is None else now
        if muted or asleep:
            if not self._sleeping_pose:
                self.enter_sleep_pose()
                return "sleep_pose"
            return None
        if self._sleeping_pose:
            self.wake_from_sleep_pose()
            return "wake_pose"
        if speaking:
            return None

        curiosity = int(behavior.get("curiosity", 50))
        expressiveness = int(behavior.get("expressiveness", 60))
        bravery = int(behavior.get("bravery", 55))
        if curiosity <= 0 and expressiveness <= 0:
            return None

        mood = str(ambient.get("mood") or "quiet")

        # 1. Sudden dramatic audio sting (loud cheer, thunder, combat drop) -> quick reaction.
        if mood == "sting" and expressiveness >= 25 and (ts - self._last_ambient_at) >= 18.0:
            self._last_ambient_at = ts
            intent = "angry" if bravery >= 80 else "surprised"
            self.play_emotion(intent)
            return f"ambient:{intent}"

        # 2. Combat or tense background music -> occasional subtle alert posture (never spammy).
        if mood in ("combat_music", "tense_music") and expressiveness >= 35 and (ts - self._last_ambient_at) >= 32.0:
            self._last_ambient_at = ts
            if mood == "combat_music":
                intent = "angry" if bravery >= 70 else ("scared" if bravery <= 30 else "attentive")
            else:
                intent = "curious" if bravery >= 50 else "scared"
            self.play_emotion(intent)
            return f"ambient:{intent}"

        # 3. Calm tavern/noir background music -> gentle, quiet rhythmic antenna bob.
        if mood == "calm_music" and curiosity >= 20:
            sway_gap = max(2.0, 5.5 - (curiosity / 30.0))
            if (ts - self._last_sway_at) >= sway_gap:
                self._last_sway_at = ts
                self._sway_phase += 1.25
                amp = 0.08 + (curiosity / 100.0) * 0.14  # softly clamped for quiet motors
                left = round(math.sin(self._sway_phase) * amp, 3)
                right = round(math.cos(self._sway_phase) * amp, 3)
                self._post("/api/move/goto", {"antennas": [left, right], "duration": 0.9})
                return "music_sway"

        # 4. Lively table chatter -> occasional subtle DoA ear perk / look toward speaker.
        if mood == "lively_table" and curiosity >= 35 and (ts - self._last_sway_at) >= 14.0:
            doa = self._get("/api/state/doa")
            if isinstance(doa, dict) and doa.get("speech_detected"):
                self._last_sway_at = ts
                angle = float(doa.get("angle") or 0.0)
                yaw = max(-0.32, min(0.32, (angle - (math.pi / 2.0)) * 0.35))
                self._post(
                    "/api/move/goto",
                    {
                        "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": -0.04, "yaw": yaw},
                        "antennas": [0.12, 0.12],
                        "duration": 0.7,
                    },
                )
                return "doa_perk"
        return None
