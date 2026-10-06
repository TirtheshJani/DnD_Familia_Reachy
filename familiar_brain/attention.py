"""Where Rocky looks, decided once per turn (design 4.7, as revised 2026-10-02).

Head tracking is the conversation app's daemon-side face tracker. It is on whenever
Rocky is awake, from the first line the brain hears, whoever that line was said to,
and off while muted: muted is asleep, and the mute cue plays sleepy. Changes go out on
the next turn that carries a preface, never on a timer.

There are no glances. Tracking used to be on only while Rocky was addressed, so that a
move_head toward the sound direction could show on the other lines. TJ wanted the face
tracking always on, and move_head is invisible while the tracker owns the head at
weight 1, so the glance went with the off state (docs/deviations.md P14).

Toggles go out only on a change: the app's MovementManager keeps its own flag and a
repeated head_tracking(true) would reset its speaking handoff.
"""
from __future__ import annotations

import logging

from familiar_brain.motion import tracking_item
from familiar_brain.state import BrainState

log = logging.getLogger("familiar.attention")


class Attention:
    def __init__(self, *, state: BrainState):
        self.state = state

    def items(self) -> list[dict]:
        want = not self.state.muted
        if want == self.state.tracking:
            return []
        self.state.tracking = want
        log.info("head tracking %s", "on" if want else "off")
        return [tracking_item(want)]
