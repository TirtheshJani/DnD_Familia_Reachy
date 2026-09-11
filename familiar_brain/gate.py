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
        now = self.clock()
        if self.state.consume_speak_now() or self._pattern.search(last):
            self.state.mark_named(now)
            return Decision.ADDRESSED
        # The window is anchored to naming, not to speech. Anchoring it to speech meant
        # every reply re-armed it, so the familiar answered the whole table forever.
        if self.state.last_named_at is not None and self.state.follow_ups < self.max_follow_ups:
            deadline = self.state.last_named_at + self.window
            spoke = self.state.last_spoke_at
            if (not self.state.window_extended and spoke is not None
                    and self.state.last_named_at <= spoke <= deadline):
                deadline = spoke + self.window   # single extension, from the first reply
                self.state.window_extended = True
            if now < deadline:
                self.state.count_follow_up()
                return Decision.ADDRESSED
            log.info("follow-up window closed at %.1f; handing back to the classifier", deadline)
        elif self.state.last_named_at is not None:
            log.info("follow-up chain hit %d; handing back to the classifier", self.max_follow_ups)
        if self.classifier is None:
            return Decision.IGNORE
        try:
            return self.classifier(segments)
        except Exception as e:  # timeout, HTTP error, unparseable
            log.warning("classifier failed (%s); defaulting to IGNORE", type(e).__name__)
            return Decision.IGNORE
