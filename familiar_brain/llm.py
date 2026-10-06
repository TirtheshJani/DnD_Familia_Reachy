from __future__ import annotations

import logging
import re
from typing import Sequence

import requests

from familiar_brain.gate import Decision

log = logging.getLogger("familiar.llm")

# Scored against twelve labelled lines of table talk on the live model. The plan's
# original wording got 7 of 12, reading Dungeon Master narration such as "You see a
# rusted iron door" as something to react to. The three way wording used until
# 2026-10-02 got 10 of 12, and both of its misses fell toward silence, which is the side
# the design wants to err on. That score is not this prompt's: the AMBIENT or IGNORE
# wording below has not been scored on the live model.
# Since 2026-10-02 it no longer offers ADDRESSED. Rocky replies only when spoken to, and
# the gate decides that by its name, the window after a naming, or Speak Now; the
# classifier only tells a moment worth reacting to from everything else.
CLASSIFY_INSTRUCTION = (
    "You are a strict filter for a small familiar named {name} that sits on the table at a "
    "game of Dungeons and Dragons. Almost everything said at the table is between the human "
    "players and the Dungeon Master, NOT to {name}. Read the recent lines, most recent last, "
    "and judge ONLY the last line.\n"
    "\n"
    "Answer with exactly one word.\n"
    "AMBIENT: the line is a burst of excitement or dismay at "
    "the table, such as a critical hit, a natural one, a death, or shared laughter.\n"
    "IGNORE: everything else. Dungeon Master narration, including narration that says \"you\" "
    "to the party. Players talking to each other or to the Dungeon Master. Rules questions "
    "asked of the table. Dice being announced. Any line where you are not certain.\n"
    "\n"
    "When in doubt answer IGNORE. Staying silent is always the safe choice."
)

# Scored against twelve labelled addressed lines on the live model. The earlier wording
# ended "Choose attentive when nothing else clearly fits" and got 6 of 12: attentive to
# praise, an opinion question, a warning and a death. One sentence naming the feelings to
# match got 9 of 12. Saying what each mood answers gets 11 of 12, and 8 of 10 on ten more
# lines it was not written against, while a roll, a greeting and a question of fact all
# stay attentive. The descriptions name the shipped persona's moods; an answer a persona
# has no gesture for is ignored and the turn goes out attentive (motion.for_addressed).
MOOD_INSTRUCTION = (
    "You choose the facial expression of a small creature that has just been spoken to at a "
    "game of Dungeons and Dragons. Read the line and answer with exactly one word from this "
    "list: {moods}.\n"
    "amused: a joke, teasing, or laughter.\n"
    "proud: praise or thanks.\n"
    "thoughtful: it is asked for its opinion, or to help think something through.\n"
    "alarmed: a warning, a sudden danger, or a shock.\n"
    "sad: bad news, a death, or an unkind word.\n"
    "attentive: a plain request, a greeting, a question of fact, or anything else.\n"
    "A word from the list that is not described above fits only when the line clearly calls for it."
)

# The reaction call. The line was not spoken to Rocky; it is usually the Dungeon Master
# narrating or a player talking. The list passed in ends with "none" and the wording
# pushes toward it, for the same reason CLASSIFY_INSTRUCTION pushes toward IGNORE: a
# creature that reacts to everything reacts to nothing. cheer was split out of amused on
# 2026-10-02 (deviation P15) and has not been scored against the live model.
REACTION_INSTRUCTION = (
    "You choose the reaction of a small creature listening at a game of Dungeons and "
    "Dragons. The line was not spoken to it; it is usually the Dungeon Master narrating or "
    "a player talking. Answer with exactly one word from this list: {moods}.\n"
    "amused: the line is funny.\n"
    "cheer: a victory or a triumph for the party.\n"
    "alarmed: a shocking twist, a sudden reveal, a betrayal.\n"
    "sad: grief, loss, a farewell, something sorrowful.\n"
    "scared: danger or a threat to the party, a monster, a trap springing.\n"
    "none: ordinary description, travel, rules talk, dice, table logistics.\n"
    "Most lines are none."
)

# First run of letters, whole word: "**Sad**" and "alarmed: a shocking" parse, while
# "not sad", "no reaction" and "Sadly, none" do not (their first word is not a mood).
_MOOD_WORD = re.compile(r"[a-z]+")


class LLMClient:
    def __init__(self, *, base_url: str, classify_slot: int, classify_timeout_s: float,
                 generate_slot: int = 0):
        self.base_url = base_url.rstrip("/")
        self.classify_slot = classify_slot
        self.classify_timeout_s = classify_timeout_s
        self.generate_slot = generate_slot

    def complete(self, messages: list[dict], max_tokens: int = 1200) -> str:
        """One non-streaming completion on the generate slot, for the session summariser.

        No timeout as tight as classify's: this runs in a background thread after the
        session has ended, so a slow answer costs nobody anything. reasoning_effort is
        none for the same reason it is on classify, see docs/deviations.md E2.

        A reply stopped at max_tokens raises rather than coming back as if whole: a summary
        or campaign memory cut off there has lost its last headings (Open threads, Loot),
        and once written it marks the session learnt, so it would never be retried.
        """
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.2,
                "stream": False, "id_slot": self.generate_slot, "reasoning_effort": "none"}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError(f"the reply was cut off at max_tokens ({max_tokens})")
        return choice["message"]["content"]

    def classify(self, segments: Sequence[str], *, persona_name: str) -> Decision:
        # Only the last line. Design 5 layer 3 calls for the last four segments, but
        # measured on labelled table talk that context scored 4/12 with 7 uninvited
        # replies, against 11/12 and none for the last line alone: any earlier segment
        # naming the familiar made the model read everything after it as one continuing
        # conversation. Layers 1 and 2 already cover "it was named" and "it just spoke".
        lines = "\n".join(f"- {s}" for s in segments[-1:])
        body = {
            "messages": [
                {"role": "system", "content": CLASSIFY_INSTRUCTION.format(name=persona_name)},
                {"role": "user", "content": lines},
            ],
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
            "id_slot": self.classify_slot,
            # Gemma 4 reasons before answering, which does not fit a one-token budget on
            # a 600 ms timeout. Measured live: 286 ms with reasoning on, 59 ms with it off.
            "reasoning_effort": "none",
        }
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.classify_timeout_s)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip().upper()
        # ADDRESSED is not on offer (see CLASSIFY_INSTRUCTION), so it does not parse
        # either: the gate then logs a failed call and stays silent.
        for d in (Decision.AMBIENT, Decision.IGNORE):
            if text.startswith(d.value):
                return d
        raise ValueError(f"unparseable classifier output: {text!r}")

    def mood(self, text: str, *, moods: tuple[str, ...], prompt: str = MOOD_INSTRUCTION) -> str | None:
        """One word from `moods` for the gesture that goes out with a turn.

        Same slot, budget and reasoning setting as classify: 59 ms measured live for one
        token with reasoning off (docs/versions.md, gate classifier table; docs/deviations.md
        E5). `prompt` selects the instruction; app.py passes MOOD_INSTRUCTION to the
        Choreographer's mood classifier for addressed turns and REACTION_INSTRUCTION to its
        reaction classifier for table lines. The first word of the answer must equal a
        listed mood, so mood keys must be single lowercase words. Returns None for anything
        else, and logs the raw answer. The keyword rules have already been tried by then
        (motion.Choreographer), so None means the default mood on an addressed turn and no
        reaction on a table line.
        """
        body = {
            "messages": [
                {"role": "system", "content": prompt.format(moods=", ".join(moods))},
                {"role": "user", "content": text},
            ],
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
            "id_slot": self.classify_slot,
            "reasoning_effort": "none",
        }
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.classify_timeout_s)
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"]
        m = _MOOD_WORD.search(answer.lower())
        word = m.group() if m else ""
        if word not in moods:
            log.info("mood answer not in list, ignored: %r", answer)
        return word if word in moods else None
