from __future__ import annotations

import logging
import re
import time
from enum import Enum
from typing import Callable, Sequence

from familiar_brain.state import BrainState

log = logging.getLogger("familiar.gate")


class Decision(str, Enum):
    ADDRESSED = "ADDRESSED"
    AMBIENT = "AMBIENT"
    IGNORE = "IGNORE"


Classifier = Callable[[Sequence[str]], Decision]


class Gate:
    def __init__(self, *, state: BrainState, name: str, aliases: Sequence[str],
                 follow_up_window_s: float, classifier: Classifier | None,
                 clock: Callable[[], float] = time.time, max_follow_ups: int = 3):
        self.state = state
        names = [name, *aliases]
        self._pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)
        self.window = follow_up_window_s
        self.classifier = classifier
        self.clock = clock
        self.max_follow_ups = max_follow_ups

    def decide(self, segments: Sequence[str]) -> Decision:
        """segments: most recent last. Only the last segment is the candidate utterance."""
        if self.state.muted:
            return Decision.IGNORE
        last = segments[-1] if segments else ""
        if self.state.consume_speak_now() or self._pattern.search(last):
            self.state.reset_follow_ups()
            return Decision.ADDRESSED
        if self.state.last_spoke_at is not None and (self.clock() - self.state.last_spoke_at) < self.window:
            # Speaking re-arms this window, so an uncapped chain means the familiar
            # answers every utterance for the rest of a session that never goes quiet.
            if self.state.follow_ups < self.max_follow_ups:
                self.state.count_follow_up()
                return Decision.ADDRESSED
            log.info("follow-up chain hit %d; handing back to the classifier", self.max_follow_ups)
        if self.classifier is None:
            return Decision.IGNORE
        try:
            return self.classifier(segments)
        except Exception as e:  # timeout, HTTP error, unparseable
            log.warning("classifier failed (%s); defaulting to IGNORE", type(e).__name__)
            return Decision.IGNORE
