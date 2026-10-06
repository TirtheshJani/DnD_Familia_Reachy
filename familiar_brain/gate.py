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


# Rocky replies only when spoken to, and players also talk ABOUT it. Live on 2026-09-16
# the one uninvited reply in seventy lines came from an 868 word segment, minutes of
# chatter the transcriber had merged, in which players explained the familiar to each
# other: "we're doing uh rocky", "What's his name, sorry? Rocky." The rules below read
# those shapes as mentions. Like the adjective rules they err towards not naming, and
# like them they are narrow, because a missed vocative costs a player repeating the name.

# Transcribed hesitations, skipped when looking at the words before the name.
_FILLERS = r"(?:\s+(?:uh|um|uhm|er|erm))*"
_MENTION_BEFORE = re.compile(
    r"(?:"
    # The name introduced as a name: "his name is Rocky", "it's called Rocky",
    # "named Rocky", "we named him Rocky", "I call him Rocky".
    r"\bname(?:'s|\s+is|\s+was)|\b(?:named|called)(?:\s+(?:him|her|it|them))?|\bcall\s+(?:him|her|it|them)"
    # "we're doing rocky": the player choosing it as a familiar. Only with a first or
    # third person subject, because "what are you doing Rocky" is a question to it.
    r"|\b(?:we're|we\s+are|i'm|i\s+am|they're|he's|she's)\s+doing"
    r"|\b(?:talk|talking|talked|speaking|ask|asking|asked)\s+about"
    r")" + _FILLERS + r"\s+$", re.IGNORECASE)
# The words above also end a question put to Rocky once the transcriber drops the comma
# before its name: "what's this called Rocky", "what were we talking about Rocky". A
# clause that opens with a question word, with the name last or before its "?", is that
# question, and recall is what a familiar with notes is for.
_CLAUSE = re.compile(r"[^.!?;:,]*$")
_CLAUSE_OPENERS = frozenset({"so", "okay", "ok", "and", "but", "hey", "oh", "um", "uh", "well",
                             "then", "wait", "sorry"})
_QUESTION_WORDS = frozenset({
    "what", "what's", "whats", "who", "who's", "whom", "whose", "where", "where's", "which",
    "how", "how's", "why", "when", "did", "didn't", "do", "don't", "does", "can", "can't",
    "could", "should", "would", "will", "shall", "is", "are", "was", "were", "have", "has",
    "had", "may", "might",
})
_NAME_ENDS_QUESTION = re.compile(r"\s*(?:\?|[.!]?\s*$)")
# "What's his name, sorry? Rocky." The name on its own answering a question about the
# familiar's name. Only his, her, its or their name, or what it is called: "what was the
# innkeeper's name? Rocky?" asks Rocky to remember. And only a bare answer, the name then
# a stop or nothing: "So what's his name? Rocky, do you know?" asks Rocky too.
_NAME_QUESTION_BEFORE = re.compile(
    r"\b(?:(?:his|her|its|their)\s+name|familiar's\s+name|call\s+(?:him|her|it|them))\b"
    r"[^.!?]{0,20}\?\s*$", re.IGNORECASE)
_BARE_ANSWER = re.compile(r"\s*(?:[.!]|$)")
# A possessive describes the familiar: "Rocky's eyes are glowing". "Rocky's turn" and
# "Rocky's go" are the exception, handing the familiar the next move.
_POSSESSIVE = re.compile(r"['\u2019]s\b(?:\s+(\w+))?", re.IGNORECASE)
_POSSESSIVE_INVITES = frozenset({"turn", "go"})


def _asks_rocky(head: str, after: str) -> bool:
    words = _CLAUSE.search(head).group(0).lower().split()
    while words and words[0] in _CLAUSE_OPENERS:
        words.pop(0)
    return bool(words) and words[0] in _QUESTION_WORDS and _NAME_ENDS_QUESTION.match(after) is not None


def _reads_as_mention(text: str, match: re.Match[str]) -> bool:
    head = text[:match.start()].replace("\u2019", "'")
    after = text[match.end():]
    if _MENTION_BEFORE.search(head) and not _asks_rocky(head, after):
        return True
    if _NAME_QUESTION_BEFORE.search(head) and _BARE_ANSWER.match(after):
        return True
    possessive = _POSSESSIVE.match(text, match.end())
    return possessive is not None and (possessive.group(1) or "").lower() not in _POSSESSIVE_INVITES


# A follow-up inside the window is a line to Rocky only if it is a line. Real follow-ups
# live have run to about ten words ("Anything else?", "and then?"); over 60 words is
# twenty seconds and more of unbroken speech, which is the transcriber gluing table
# chatter together, as the 868 word segment was.
_MAX_FOLLOW_UP_WORDS = 60
# In such a run-on a naming counts only near the end, because the line just said to the
# familiar is the last thing the transcriber glued on. The live run-on had 64 words after
# its last "rocky". Words, not sentences, because transcripts often come back
# unpunctuated. 25 words is about ten seconds of speech. A segment of 60 words or fewer
# is one utterance, so a long recap request that starts with the name still counts.
_MAX_WORDS_AFTER_NAMING = 25


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
        run_on = len(text.split()) > _MAX_FOLLOW_UP_WORDS
        for match in self._pattern.finditer(text):
            if run_on and len(text[match.end():].split()) > _MAX_WORDS_AFTER_NAMING:
                log.info("%r is too far from the end of the segment to be a naming", match.group(0))
            elif _reads_as_adjective(text, match):
                log.info("%r reads as an ordinary word here, not a naming", match.group(0))
            elif _reads_as_mention(text, match):
                log.info("%r is talked about here, not spoken to", match.group(0))
            else:
                return True
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
            if now >= deadline:
                log.info("follow-up window closed at %.1f; handing back to the classifier", deadline)
            elif len(last.split()) > _MAX_FOLLOW_UP_WORDS:
                # Not counted as a follow-up either, so the chatter does not use one up.
                log.info("a %d word segment is table chatter, not a follow-up", len(last.split()))
            else:
                self.state.count_follow_up()
                return Decision.ADDRESSED
        elif self.state.last_named_at is not None:
            log.info("follow-up chain hit %d; handing back to the classifier", self.max_follow_ups)
        if self.classifier is None:
            return Decision.IGNORE
        try:
            decision = self.classifier(segments)
        except Exception as e:  # timeout, HTTP error, unparseable
            log.warning("classifier failed (%s); defaulting to IGNORE", type(e).__name__)
            return Decision.IGNORE
        # Speech needs a naming, the window after one, or Speak Now. The classifier used
        # to answer ADDRESSED for lines "only Rocky could be expected to answer", which at
        # a table of players asking each other questions is most of them, and it so often
        # timed out live that it could not be relied on to say so either.
        if decision is Decision.ADDRESSED:
            log.info("classifier said ADDRESSED for an unnamed line; treating it as IGNORE")
            return Decision.IGNORE
        return decision
