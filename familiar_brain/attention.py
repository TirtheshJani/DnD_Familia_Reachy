"""Where Rocky looks, decided once per turn (design 4.7).

Head tracking is the conversation app's daemon-side face tracker. It is on while
Rocky is being spoken to and for a linger after the unextended follow-up window
(last_named_at + window + linger; the gate's one-off extension is not mirrored), and
off otherwise, because move_head is invisible while the tracker owns the head at
weight 1. Changes go out on the next turn, never on a timer. While it is off, an IGNORE or AMBIENT line earns a glance toward the sound
direction the daemon last measured, rate limited so Rocky watches the table rather
than twitching at every line.

Toggles go out only on a change: the app's MovementManager keeps its own flag and a
repeated head_tracking(true) would reset its speaking handoff.
"""
from __future__ import annotations

import logging
from typing import Callable

from familiar_brain.gate import Decision
from familiar_brain.motion import glance_item, tracking_item
from familiar_brain.robot import direction_from_angle
from familiar_brain.state import BrainState

log = logging.getLogger("familiar.attention")

DoaReader = Callable[[], tuple[float, bool] | None]


class Attention:
    def __init__(self, *, state: BrainState, doa: DoaReader, follow_up_window_s: float,
                 linger_s: float, glance_gap_s: float):
        self.state = state
        self.doa = doa
        self.window = follow_up_window_s
        self.linger = linger_s
        self.glance_gap = glance_gap_s

    def wants_tracking(self, decision: Decision, now: float) -> bool:
        if self.state.muted:
            return False
        if decision is Decision.ADDRESSED:
            return True
        # The linger keeps tracking on; it never switches it on. After a mute has turned
        # tracking off, a line nobody spoke to Rocky must not turn it back on.
        named = self.state.last_named_at
        return self.state.tracking and named is not None and now < named + self.window + self.linger

    def items(self, decision: Decision, now: float) -> list[dict]:
        out: list[dict] = []
        want = self.wants_tracking(decision, now)
        if want != self.state.tracking:
            self.state.tracking = want
            log.info("head tracking %s", "on" if want else "off")
            out.append(tracking_item(want))
        # Past this point the turn is IGNORE or AMBIENT: an unmuted ADDRESSED turn always
        # wants tracking.
        if want or self.state.muted:
            return out
        if now - self.state.last_glance_at < self.glance_gap:
            return out
        reading = self.doa()
        if reading is None:
            return out
        angle, _speech = reading
        direction = direction_from_angle(angle)
        self.state.last_glance_at = now
        log.info("glance %s (doa %.2f rad)", direction, angle)
        out.append(glance_item(direction))
        return out
