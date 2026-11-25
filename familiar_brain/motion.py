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

from familiar_brain.behavior import BehaviorSnapshot, scale_reaction_gaps, tilt_threat_mood
from familiar_brain.persona import Persona
from familiar_brain.responses_api import call_item

log = logging.getLogger("familiar.motion")

MoodClassifier = Callable[[str], str | None]

# The reactions Rocky may show to a line not spoken to it. The reaction call may also
# answer "none"; anything outside this tuple is no gesture. cheer is a victory for the
# party (success), split from amused (happy) so a slain dragon does not get the grin a
# joke gets; angry, joyful, and curious widen the living-creature palette.
REACTION_MOODS: tuple[str, ...] = ("amused", "cheer", "alarmed", "sad", "scared")

# Lines shorter than this never reach the reaction call: "pass the chips" is not
# narration, and the keyword rules already cover short outbursts. An AMBIENT line skips
# this floor (Choreographer.for_table).
MIN_REACTION_WORDS = 5

# What an AMBIENT line with no rule hit plays when the reaction call fails or answers
# none. The gate answers AMBIENT only for a burst of excitement or dismay, and does not
# say which. Surprised reads as engaged for a cheer and a groan alike, where amused would
# grin at a friend going down and sad would mope at a natural twenty.
AMBIENT_FALLBACK_MOOD = "alarmed"

# The lean-in when combat starts. It spends the gesture gap but not the reaction gap: a
# lean-in is worth less than the first crit of the fight it announces.
LEAN_IN_MOOD = "attentive"

# Laughter, shared by both rule sets: "hahaha", "ha ha", "ahaha", but not "Aha!".
_LAUGHTER = r"a?(ha ?){2,}|hilarious"

# A phrase that is the whole of what was said, or ends its sentence: what separates the
# exclamation "No way!" from "no way to get across", and "we did it" from "we did it wrong".
_ENDS = r"(?=\s*([,.!?]|$))"
_STARTS = r"(^|[.!?,]\s*)"
# Stricter: the phrase ends its sentence, not just its clause. "Oh no, how are you?" is
# a greeting and "Holy crap, it's ten o'clock" a remark about the time.
_ALONE = r"(?=\s*([.!?]|$))"

# Design section 5: a non-addressed line may emit one gesture chosen by a small keyword
# rule set. These run before the reaction call and cost nothing, which matters: live on
# 2026-09-16 the reaction call answered none on 49 table lines, timed out on 4 and chose
# a reaction once, so these rules are nearly all the expression Rocky has between
# replies. The first hit wins, so the specific rules (the dice, a friend down, laughter)
# come before the general exclamations: "oh no, he went down" is scared, not surprised.
TABLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(nat(ural)? ?(one|1)|crit(ical)? ?fail(ure|ed|s)?|fumbl(e|ed|es|ing))\b", re.I), "alarmed"),
    # "rolled a twenty" for the transcripts that drop the "nat". Not "crits": "he crits
    # you for 22" is the monster's, and amused would grin at it.
    (re.compile(r"\b(nat(ural)? ?(twenty|20)|crit(ical)?( hit)?|rolled an? (twenty|20))\b", re.I), "amused"),
    (re.compile(r"\b(went down|(he|she|they)('s|'re| is| are) (down|dead|dying)|get (him|her|them) up|unconscious)\b", re.I), "scared"),
    (re.compile(rf"\b({_LAUGHTER})\b", re.I), "amused"),
    # The transcriber's own note of laughter. Not a bare "laughs": the DM narrates that.
    (re.compile(r"[\[(]\s*laugh(s|ing|ter)?\s*[\])]", re.I), "amused"),
    (re.compile(r"\b(funeral|farewell|grief|mourn(s|ing)?|weep(s|ing)?|in tears|passed away|rest in peace"
                r"|heartbreaking|that('s| is) (so )?sad|(he|she|they) didn't make it)\b", re.I), "sad"),
    # Shock. "Oh my" needs its god after it ("Oh my, the goddess"); "oh no", "no way",
    # "holy crap", "are you kidding" and the oath after "what the" must end their
    # sentence, so "What the heck is a sennight?" is a lore question. Live, every "oh
    # no" in non-addressed lines was small talk ("Oh no, how are you?"). "No no no" is
    # an exclamation wherever it falls, which matters as transcripts drop the "!".
    (re.compile(rf"\boh no( no)*\b{_ALONE}|{_STARTS}no way\b{_ALONE}|\boh my (god|gods|gosh|goodness)\b"
                rf"|\bwhat the (hell|heck|fuck|frick)( (was|is) that| just happened)?{_ALONE}"
                rf"|\b(are you|you're) (kidding|joking)( me)?{_ALONE}"
                rf"|\bholy (shit|crap|cow|moly){_ALONE}|\bno[,.!]? no[,.!]? no\b", re.I), "alarmed"),
    # Victory. Only an exclaimed or drawn out "let's go" ("let's go." moves the game on),
    # a "yes" that is a shout rather than an answer (shouted twice needs no "!"), and "we
    # did it" or "we won" that end their sentence. "Killed it" alone is left out: here it
    # is as likely the wolf that killed the horse as the party that killed the wolf, so
    # only "we killed" or "we got" with a "!" counts. "Victory" and its kin only as a cry
    # of their own: "Victory Road" is a place and "roar in victory" may be the orcs'.
    (re.compile(rf"\blet'?s go+!|\blet'?s goo+\b|{_STARTS}((hell|heck|fuck) )?yes+( yes+)*!"
                rf"|{_STARTS}yes+( yes+)+\b|{_STARTS}(hell|heck|fuck) yes+{_ENDS}"
                rf"|\bwe (did it|won|beat (him|her|it|them))\b(?!'){_ENDS}"
                r"|\bwe (killed|slew|got) (him|her|it|them|the \w+)\s*!"
                rf"|{_STARTS}(huzzah|hurrah|hooray|victory)\b{_ENDS}"
                r"|\b(nailed it|woo+ ?hoo+)\b|\bwoo+!", re.I), "cheer"),
    # Betrayal, outrage, or fierce hostility in DM narration or table reaction.
    (re.compile(r"\b(betray(ed|s|al)?|traitor|stabbed (us|him|her|them) in the back"
                r"|how dare (you|he|she|they)|double[- ]cross(ed)?)\b", re.I), "angry"),
    # Joyful wonder or treasure delight.
    (re.compile(r"\b(marvelous|magnificent|treasure hoard|jackpot|that's amazing)\b", re.I), "joyful"),
    # Curious intrigue on secret doors or arcane clues.
    (re.compile(r"\b(secret (door|passage|compartment)|hidden door|mysterious glowing|strange rune)\b", re.I), "curious"),
    # Combat starts: Rocky leans in. This plays nothing over speech, as no reply follows.
    # Not in a question: "do I roll initiative or is this a surprise round?" is rules talk.
    (re.compile(r"\broll (for )?initiative\b(?![^.!]*\?)", re.I), LEAN_IN_MOOD),
)

# Table rules read only the end of a segment. The transcriber merges minutes of table
# talk into one segment, and the line that just happened is the last thing in it; the
# gate looks for a naming only this near the end of a run-on, for the same reason.
TABLE_TAIL_WORDS = 25

# A warning shouted at Rocky. "Look out" and "watch out" count only where the warning ends,
# at a stop, a comma, the end of the line or "behind": "keep a look out for guards" and
# "look out the window" are plain requests at a heist table. "Behind us" counts unless the
# line puts or leaves something there ("let's put that behind us").
_WARNING = (r"(?<!keep a )\b(look|watch) out\b(?=\s*([,.!?]|$|behind\b))"
            r"|\bdanger\b"
            r"|^(?!.*\b(put|putting|leave|leaving|left)\b).*\bbehind us\b")

# A greeting or a goodbye, but only when it is the whole line: a word of address either
# side ("Hello, Rocky", "Rocky, goodnight") and, for a greeting, a "how are you". "Hello
# Rocky, roll me a d20" is a roll. "Hey" is not a greeting here: it calls Rocky's
# attention before nearly every request, and a welcoming wave on each would be the
# constant emoting TJ asked about on 2026-09-16. A goodbye waves rather than plays
# sleepy, since sleepy is the mute cue's gesture and would read as Rocky going to sleep.
_GREETING = (r"^\W*(\w+\W+)?"
             r"((hi|hello|hiya|howdy|good (morning|afternoon|evening)|welcome( back)?)\b(\W+\w+)?"
             r"(\W+(how are you|how's it going|how are things))?"
             r"|(good ?night|goodbye|bye( bye)?|see you( (later|soon|next (time|week|session)))?|see ya|night night)\b"
             r"(\W+\w+)?)\W*$")

# ADDRESSED turns. Any rule hit wins over the one-token mood call, because live the call
# answered attentive to praise, an opinion question and a warning alike; a roll wins too,
# as the commonest request and one the call answered attentive anyway, so it costs no
# round trip. Winning means a misfire is a wrong gesture the model might have avoided, so
# each rule holds only phrases that carry their mood wherever they appear: a bare "dragon"
# is the game's name and its subject, not a threat, and is left to the call. Order
# matters; a warning outranks a request for an opinion in the same line.
ADDRESSED_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(_WARNING, re.I), "alarmed"),
    (re.compile(_GREETING, re.I), "awake"),
    (re.compile(rf"\b({_LAUGHTER}|(good|great|funny) joke)\b", re.I), "amused"),
    (re.compile(r"\b(good (boy|girl|job)|well done|clever(?! by half))\b", re.I), "proud"),
    (re.compile(r"\b(what do you (think|make|reckon)|thoughts|opinion|help me think|think with me)\b", re.I), "thoughtful"),
    (re.compile(r"\broll\b", re.I), "attentive"),
)
DEFAULT_ADDRESSED_MOOD = "attentive"


def table_mood(text: str) -> str | None:
    text = " ".join(text.split()[-TABLE_TAIL_WORDS:])
    for pattern, mood in TABLE_RULES:
        if pattern.search(text):
            return mood
    return None


def addressed_mood(text: str) -> str | None:
    for pattern, mood in ADDRESSED_RULES:
        if pattern.search(text):
            return mood
    return None


_DEEP_NIGHT_ON = re.compile(
    r"^\s*(?:(?:it['\u2019]s|it\s+is)\s+)?deep\s+night\b(?!\s+is\s+over)(?:[\s,!.:].*)?$",
    re.IGNORECASE,
)
_DEEP_NIGHT_OFF = re.compile(
    r"^\s*(?:deep\s+night\s+is\s+over|good\s+morning)(?:[\s,!.:].*)?$",
    re.IGNORECASE,
)


def sleep_transition(text: str) -> bool | None:
    """Return True to enter Deep Night, False to wake from it, or None if not a declaration."""
    stripped = text.strip()
    if not stripped or stripped.endswith("?"):
        return None
    if _DEEP_NIGHT_OFF.match(stripped):
        return False
    if _DEEP_NIGHT_ON.match(stripped):
        return True
    return None


def tracking_item(enabled: bool) -> dict:
    return call_item("head_tracking", {"enabled": bool(enabled)})


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
                 reaction_classifier: MoodClassifier | None = None,
                 behavior_provider: Callable[[], BehaviorSnapshot] | None = None):
        self.persona = persona
        self.min_gap_s = min_gap_s
        self.reaction_gap_s = reaction_gap_s
        self.clock = clock
        self.mood_classifier = mood_classifier
        self.reaction_classifier = reaction_classifier
        self.behavior_provider = behavior_provider
        self.last_at: float = float("-inf")
        self.last_reaction_at: float = float("-inf")

    def _effective_gaps(self) -> tuple[float, float] | None:
        if self.behavior_provider is None:
            return (self.min_gap_s, self.reaction_gap_s)
        return scale_reaction_gaps(
            self.min_gap_s, self.reaction_gap_s, self.behavior_provider().expressiveness
        )

    def _apply_bravery(self, mood: str | None) -> str | None:
        if self.behavior_provider is None or mood is None:
            return mood
        tilted = tilt_threat_mood(mood, self.behavior_provider().bravery)
        return tilted if tilted in self.persona.gestures else mood

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
        gaps = self._effective_gaps()
        if gaps is None:
            return []
        min_gap_s, _ = gaps
        if mood is None or now - self.last_at < min_gap_s:
            return []
        item = self.gesture(mood)
        if item is None:
            return []
        self.last_at = now
        return [item]

    def for_addressed(self, text: str, now: float) -> list[dict]:
        """The gesture for a reply whose line carries a mood, or nothing."""
        gaps = self._effective_gaps()
        if gaps is None:
            return []
        min_gap_s, _ = gaps
        if now - self.last_at < min_gap_s:
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

    def for_table(self, text: str, now: float, *, ambient: bool = False) -> list[dict]:
        """A reaction to a line not spoken to Rocky, or nothing."""
        gaps = self._effective_gaps()
        if gaps is None:
            return []
        min_gap_s, reaction_gap_s = gaps
        if now - self.last_at < min_gap_s or now - self.last_reaction_at < reaction_gap_s:
            return []
        mood = table_mood(text)
        if mood not in self.persona.gestures:
            mood = None
        answer = None
        if (mood is None and self.reaction_classifier is not None
                and (ambient or len(text.split()) >= MIN_REACTION_WORDS)):
            try:
                answer = self.reaction_classifier(text)
            except Exception as e:  # never let a reaction cost anything
                log.warning("reaction classifier failed (%s); no reaction", type(e).__name__)
            mood = answer if answer in REACTION_MOODS and answer in self.persona.gestures else None
            log.info("reaction call answered %r", answer)
        if mood is None and ambient and answer != "none":
            log.info("ambient burst with no reaction; falling back to %s", AMBIENT_FALLBACK_MOOD)
            mood = AMBIENT_FALLBACK_MOOD
        mood = self._apply_bravery(mood)
        items = self._gated(mood, now)
        if items and mood != LEAN_IN_MOOD:
            self.last_reaction_at = now
        return items

    def cue(self, mood: str, now: float) -> list[dict]:
        item = self.gesture(mood)
        if item is None:
            return []
        self.last_at = now
        return [item]
