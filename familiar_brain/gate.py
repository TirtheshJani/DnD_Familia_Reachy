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

# A persona name can also be an ordinary word. "rocky" is an adjective, and a Dungeon
# Master says "a rocky outcrop" or "the path is rocky" far more often than anyone talks
# to Rocky, so a bare word match would wake it on narration. A match that reads as the
# adjective is not a naming. That errs towards not naming on purpose: such a line still
# falls through to the follow-up window and then the classifier. The classifier is not a
# safety net to lean on, though. It often times out, and the transcriber drops the comma
# after a name, so "Rocky pass me the dice" and "Me too Rocky" read as adjectives and
# Rocky stayed silent although named. The rules below are narrowed for those two shapes.

# A determiner directly before the word: "a rocky", "the rocky pass".
_DETERMINERS = frozenset({"a", "an", "the"})
# An intensifier directly before the word, but only when another word follows in the
# same clause: "too rocky to cross" is the adjective, while "Me too Rocky" is a naming.
_INTENSIFIERS = frozenset({"very", "too", "quite", "pretty", "rather", "fairly"})
# A copula directly before the word, but only when the word also ends the clause:
# "the path is rocky." is the adjective, while "is Rocky awake?" and "where is Rocky?"
# are about the familiar, which is why a question mark does not end the clause here.
_COPULAS = frozenset({"is", "was", "are", "were", "be", "been", "looks", "seems", "gets", "got",
                      "getting", "it's", "that's", "they're"})
# A terrain-like noun directly after the word: "rocky ground", "rocky terrain".
_TERRAIN = frozenset({
    "terrain", "ground", "path", "paths", "road", "roads", "shore", "shores", "beach", "coast",
    "cliff", "cliffs", "outcrop", "outcrops", "slope", "slopes", "hill", "hills", "ledge",
    "ledges", "soil", "walls", "cave", "caves", "tunnel", "island", "crag", "crags",
})
# These also follow the adjective ("a rocky start", "the rocky pass"), but they double as
# imperative verbs. With no word before the name in its sentence, "Rocky pass me the dice"
# is an order to the familiar, so there only the terrain nouns above count.
_TERRAIN_OR_VERB = frozenset({"pass", "start", "trail", "trails", "patch", "floor"})
_WORD_BEFORE = re.compile(r"([\w'\u2019]+)\s+$")
_WORD_AFTER = re.compile(r"\s+(\w+)")
_ENDS_CLAUSE = re.compile(r"\s*(?:[.!,;:]|$)")
# The start of a sentence, or of one opened by a greeting: "Hey Rocky pass me the dice".
_SENTENCE_START = re.compile(r"(?:^|[.!?])\s*(?:(?:hey|hi|okay|ok|oh|yo)[,!]?\s+)?$", re.IGNORECASE)


def _reads_as_adjective(text: str, match: re.Match[str]) -> bool:
    # Only a single word has an adjective reading. A phrase alias such as "little one"
    # is already a noun phrase, and "the little one" is how people say it.
    if len(match.group(0).split()) > 1:
        return False
    head = text[:match.start()]
    before = _WORD_BEFORE.search(head)
    before_word = before.group(1).lower().replace("\u2019", "'") if before else ""
    after = text[match.end():]
    after_word = _WORD_AFTER.match(after)
    nouns = _TERRAIN if _SENTENCE_START.search(head) else _TERRAIN | _TERRAIN_OR_VERB
    return (before_word in _DETERMINERS
            or (before_word in _INTENSIFIERS and after_word is not None)
            or (before_word in _COPULAS and _ENDS_CLAUSE.match(after) is not None)
            or (after_word is not None and after_word.group(1).lower() in nouns))


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

    def _named(self, text: str) -> bool:
        """In "Rocky, is the path rocky?" one naming use is enough to address the familiar."""
        for match in self._pattern.finditer(text):
            if not _reads_as_adjective(text, match):
                return True
            log.info("%r reads as an ordinary word here, not a naming", match.group(0))
        return False

    def decide(self, segments: Sequence[str]) -> Decision:
        """segments: most recent last. Only the last segment is the candidate utterance."""
        if self.state.muted:
            return Decision.IGNORE
        last = segments[-1] if segments else ""
        now = self.clock()
        if self.state.consume_speak_now() or self._named(last):
            self.state.mark_named(now)
            return Decision.ADDRESSED
        # The window is anchored to naming, not to speech. Anchoring it to speech meant
        # every reply re-armed it, so the familiar answered the whole table forever.
        if self.state.last_named_at is not None and self.state.follow_ups < self.max_follow_ups:
            named = self.state.last_named_at
            spoke = self.state.last_spoke_at
            if (self.state.window_extension_s is None and spoke is not None
                    and named <= spoke <= named + self.window):
                # Single extension, from the first reply. It is stored so that it holds
                # on every later turn until the familiar is named again.
                self.state.window_extension_s = spoke - named
            deadline = named + self.window + (self.state.window_extension_s or 0.0)
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
