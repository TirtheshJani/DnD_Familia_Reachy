from __future__ import annotations

import threading
import time


class BrainState:
    """Process-wide mutable state shared by HTTP handlers and the gate."""

    def __init__(self):
        self._lock = threading.Lock()
        self.muted = False
        self.logging_enabled = True
        self._speak_now_armed = False
        self.last_spoke_at: float | None = None
        self.recent_segments: list[str] = []

    def arm_speak_now(self) -> None:
        with self._lock:
            self._speak_now_armed = True

    def consume_speak_now(self) -> bool:
        with self._lock:
            armed, self._speak_now_armed = self._speak_now_armed, False
            return armed

    def mark_spoke(self, at: float | None = None) -> None:
        self.last_spoke_at = time.time() if at is None else at

    def remember(self, segment: str, keep: int = 4) -> None:
        self.recent_segments = (self.recent_segments + [segment])[-keep:]
