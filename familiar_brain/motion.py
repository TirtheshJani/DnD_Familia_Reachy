"""Expression: which robot-side gesture goes out with a turn.

The brain cannot move the robot between turns. Its only channel to the motors is the
Responses stream speech-to-speech relays, so every gesture rides on a turn as a
function_call item for a tool the conversation app owns: play_emotion, move_head,
head_tracking. The app runs the call and sends back its function_call_output
(docs/protocol-notes.md section 4). Those three set needs_response=False in app 1.0.1,
so a gesture never costs a second generation (plan 3, "What the evidence established").

Two kinds of gesture. An ADDRESSED turn carries the mood Rocky answers in. A turn not
addressed to Rocky (IGNORE or AMBIENT) may carry a reaction to what was said: the
Dungeon Master's twist, a death, a joke. The gate classes narration as IGNORE on
purpose (deviation E8), so reactions cannot hang off AMBIENT alone.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable

from familiar_brain.persona import Persona
from familiar_brain.responses_api import call_item

log = logging.getLogger("familiar.motion")

MoodClassifier = Callable[[str], str | None]

# The reactions Rocky may show to a line not spoken to it. The reaction call may also
# answer "none"; anything outside this tuple is no gesture.
REACTION_MOODS: tuple[str, ...] = ("amused", "alarmed", "sad", "scared")

# Lines shorter than this never reach the reaction call: "pass the chips" is not
# narration, and the keyword rules already cover short outbursts.
MIN_REACTION_WORDS = 5

# Laughter, shared by both rule sets: "hahaha", "ha ha", "ahaha", but not "Aha!".
_LAUGHTER = r"a?(ha ?){2,}|hilarious"

# Design section 5: a non-addressed line may emit one gesture chosen by a small keyword
# rule set. These run before the reaction call and cost nothing.
TABLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(nat(ural)? ?(one|1)|crit(ical)? ?fail(ure|ed|s)?|fumbl(e|ed|es|ing))\b", re.I), "alarmed"),
    (re.compile(r"\b(nat(ural)? ?(twenty|20)|crit(ical)?( hit)?)\b", re.I), "amused"),
    (re.compile(r"\b(went down|(he|she|they)('s|'re| is| are) (down|dead|dying)|get (him|her|them) up|unconscious)\b", re.I), "scared"),
    (re.compile(rf"\b({_LAUGHTER})\b", re.I), "amused"),
    (re.compile(r"\b(funeral|farewell|grief|mourn(s|ing)?|weep(s|ing)?|in tears|passed away)\b", re.I), "sad"),
)

# A warning shouted at Rocky. "Look out" and "watch out" count only where the warning ends,
# at a stop, a comma, the end of the line or "behind": "keep a look out for guards" and
# "look out the window" are plain requests at a heist table. "Behind us" counts unless the
# line puts or leaves something there ("let's put that behind us").
_WARNING = (r"(?<!keep a )\b(look|watch) out\b(?=\s*([,.!?]|$|behind\b))"
            r"|\bdanger\b"
            r"|^(?!.*\b(put|putting|leave|leaving|left)\b).*\bbehind us\b")

# ADDRESSED turns. Any rule hit wins over the one-token mood call, because live the call
# answered attentive to praise, an opinion question and a warning alike; a roll wins too,
# as the commonest request and one the call answered attentive anyway, so it costs no
# round trip. Winning means a misfire is a wrong gesture the model might have avoided, so
# each rule holds only phrases that carry their mood wherever they appear: a bare "dragon"
# is the game's name and its subject, not a threat, and is left to the call. Order
# matters; a warning outranks a request for an opinion in the same line.
ADDRESSED_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(_WARNING, re.I), "alarmed"),
    (re.compile(rf"\b({_LAUGHTER}|(good|great|funny) joke)\b", re.I), "amused"),
    (re.compile(r"\b(good (boy|girl|job)|well done|clever(?! by half))\b", re.I), "proud"),
    (re.compile(r"\b(what do you (think|make|reckon)|thoughts|opinion|help me think|think with me)\b", re.I), "thoughtful"),
    (re.compile(r"\broll\b", re.I), "attentive"),
)
DEFAULT_ADDRESSED_MOOD = "attentive"


def table_mood(text: str) -> str | None:
    for pattern, mood in TABLE_RULES:
        if pattern.search(text):
            return mood
    return None


def addressed_mood(text: str) -> str | None:
    for pattern, mood in ADDRESSED_RULES:
        if pattern.search(text):
            return mood
    return None


def tracking_item(enabled: bool) -> dict:
    return call_item("head_tracking", {"enabled": bool(enabled)})


def glance_item(direction: str) -> dict:
    return call_item("move_head", {"direction": direction})


class Choreographer:
    """Turns a mood into at most one gesture item per turn, rate limited.

    Emotions queue on the robot and each plays for a few seconds; a rapid exchange
    without a gap would pile them up and Rocky would still be emoting a minute later.
    Reactions to the table have a second, longer gap of their own so a long description
    earns one look, not one per sentence. Cues (mute, unmute) bypass both gaps because
    they are deliberate state changes.
    """

    def __init__(self, persona: Persona, *, min_gap_s: float, reaction_gap_s: float,
                 clock: Callable[[], float], mood_classifier: MoodClassifier | None = None,
                 reaction_classifier: MoodClassifier | None = None):
        self.persona = persona
        self.min_gap_s = min_gap_s
        self.reaction_gap_s = reaction_gap_s
        self.clock = clock
        self.mood_classifier = mood_classifier
        self.reaction_classifier = reaction_classifier
        self.last_at: float = float("-inf")
        self.last_reaction_at: float = float("-inf")

    def gesture(self, mood: str) -> dict | None:
        spec = self.persona.gestures.get(mood)
        if spec is None:
            log.info("no gesture configured for mood %s", mood)
            return None
        tool, arg = spec
        args = {"emotion": arg} if tool == "play_emotion" else {"direction": arg}
        log.info("gesture mood=%s -> %s %s", mood, tool, json.dumps(args))
        return call_item(tool, args)

    def _gated(self, mood: str | None, now: float) -> list[dict]:
        if mood is None or now - self.last_at < self.min_gap_s:
            return []
        item = self.gesture(mood)
        if item is None:
            return []
        self.last_at = now
        return [item]

    def for_addressed(self, text: str, now: float) -> list[dict]:
        """The gesture for a reply whose line carries a mood, or nothing.

        A plain reply gets no emotion. Every addressed turn used to queue attentive, and on
        the robot attentive1 runs 4.28 s from a start 17.8 degrees off neutral, so it played
        over the first 1.3 to 2.2 s of speech. Live on 2026-09-16 TJ asked Rocky "what's
        with the weird gestures and emotes that you have that go on constantly?". Rules
        first: a hit for a mood with a gesture configured decides and skips the mood call.
        Otherwise the call, if there is one, may name a mood; the default mood, no answer
        or a mood with no gesture is no gesture, and spends no gap.
        """
        if now - self.last_at < self.min_gap_s:
            return []
        mood = addressed_mood(text)
        if mood not in self.persona.gestures and self.mood_classifier is not None:
            try:
                mood = self.mood_classifier(text)
            except Exception as e:  # a dead classify slot must not cost the reply
                log.warning("mood classifier failed (%s); no gesture", type(e).__name__)
                mood = None
            log.info("mood call answered %r", mood)
        if mood == DEFAULT_ADDRESSED_MOOD or mood not in self.persona.gestures:
            return []
        return self._gated(mood, now)

    def for_table(self, text: str, now: float) -> list[dict]:
        """A reaction to a line not spoken to Rocky, or nothing.

        Both gaps are checked before any call is made, so a rate-limited turn costs no
        round trip. Rules first; the one-token reaction call only for lines long enough
        to be narration and only when no rule matched.
        """
        if now - self.last_at < self.min_gap_s or now - self.last_reaction_at < self.reaction_gap_s:
            return []
        mood = table_mood(text)
        if (mood is None and self.reaction_classifier is not None
                and len(text.split()) >= MIN_REACTION_WORDS):
            try:
                answer = self.reaction_classifier(text)
            except Exception as e:  # never let a reaction cost anything
                log.warning("reaction classifier failed (%s); no reaction", type(e).__name__)
                answer = None
            mood = answer if answer in REACTION_MOODS else None
            log.info("reaction call answered %r", answer)
        items = self._gated(mood, now)
        if items:
            self.last_reaction_at = now
        return items

    def cue(self, mood: str, now: float) -> list[dict]:
        item = self.gesture(mood)
        if item is None:
            return []
        self.last_at = now
        return [item]
