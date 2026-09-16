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
        # When the familiar was last named, aliased, or armed by Speak Now. The
        # follow-up window hangs off this rather than off last_spoke_at, because
        # anchoring it to speech let every reply re-arm it. See docs/deviations.md E6.
        self.last_named_at: float | None = None
        # The first reply after being named may push the window out once. After that
        # the deadline is fixed until somebody names the familiar again.
        self.window_extended = False
        # How many turns in a row the follow-up window has granted since the familiar
        # was last addressed outright. A cap on top of the fixed deadline, so a rapid
        # exchange inside one window still hands back to the classifier.
        self.follow_ups = 0
        # A gesture that must go out with the next response regardless of rate limits:
        # the mute and unmute cues. Set by handlers, consumed by the responses route.
        self._pending_cue: str | None = None
        # What the brain last told the robot about head tracking. Toggles go out only
        # on a change, so the app's MovementManager is never told the same thing twice.
        self.tracking = False
        self.last_glance_at: float = float("-inf")
        # Reply mode, see modes.py. "familiar" is the two-sentence table default.
        self.mode = "familiar"
        self.mode_set_at: float | None = None

    def arm_speak_now(self) -> None:
        with self._lock:
            self._speak_now_armed = True

    def consume_speak_now(self) -> bool:
        with self._lock:
            armed, self._speak_now_armed = self._speak_now_armed, False
            return armed

    def set_cue(self, mood: str) -> None:
        with self._lock:
            self._pending_cue = mood

    def consume_cue(self) -> str | None:
        with self._lock:
            cue, self._pending_cue = self._pending_cue, None
            return cue

    def mark_spoke(self, at: float | None = None) -> None:
        self.last_spoke_at = time.time() if at is None else at

    def mark_named(self, at: float) -> None:
        """A fresh conversation: reopen the window, allow one extension, reset the cap."""
        self.last_named_at = at
        self.window_extended = False
        self.reset_follow_ups()

    def remember(self, segment: str, keep: int = 4) -> None:
        self.recent_segments = (self.recent_segments + [segment])[-keep:]

    def reset_follow_ups(self) -> None:
        self.follow_ups = 0

    def count_follow_up(self) -> int:
        self.follow_ups += 1
        return self.follow_ups
