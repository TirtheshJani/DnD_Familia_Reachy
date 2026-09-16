from __future__ import annotations

import logging
import re
from typing import Sequence

import requests

from familiar_brain.gate import Decision

log = logging.getLogger("familiar.llm")

# Scored against twelve labelled lines of table talk on the live model. The plan's
# original wording got 7 of 12, reading Dungeon Master narration such as "You see a
# rusted iron door" as something to react to. This wording gets 10 of 12, and both of
# its misses fall toward silence, which is the side the design wants to err on.
CLASSIFY_INSTRUCTION = (
    "You are a strict filter for a small familiar named {name} that sits on the table at a "
    "game of Dungeons and Dragons. Almost everything said at the table is between the human "
    "players and the Dungeon Master, NOT to {name}. Read the recent lines, most recent last, "
    "and judge ONLY the last line.\n"
    "\n"
    "Answer with exactly one word.\n"
    "ADDRESSED: the last line is unmistakably spoken TO {name}. It names {name}, or it is a "
    "direct question or command that only {name} could be expected to answer.\n"
    "AMBIENT: nobody is talking to {name}, but the line is a burst of excitement or dismay at "
    "the table, such as a critical hit, a natural one, a death, or shared laughter.\n"
    "IGNORE: everything else. Dungeon Master narration, including narration that says \"you\" "
    "to the party. Players talking to each other or to the Dungeon Master. Rules questions "
    "asked of the table. Dice being announced. Any line where you are not certain.\n"
    "\n"
    "When in doubt answer IGNORE. Staying silent is always the safe choice."
)

MOOD_INSTRUCTION = (
    "You choose the facial expression of a small creature that has just been spoken to at a "
    "game of Dungeons and Dragons. Read the line and answer with exactly one word from this "
    "list: {moods}. Choose attentive when nothing else clearly fits."
)

# The reaction call. The line was not spoken to Rocky; it is usually the Dungeon Master
# narrating or a player talking. The list passed in ends with "none" and the wording
# pushes toward it, for the same reason CLASSIFY_INSTRUCTION pushes toward IGNORE: a
# creature that reacts to everything reacts to nothing.
REACTION_INSTRUCTION = (
    "You choose the reaction of a small creature listening at a game of Dungeons and "
    "Dragons. The line was not spoken to it; it is usually the Dungeon Master narrating or "
    "a player talking. Answer with exactly one word from this list: {moods}.\n"
    "amused: the line is funny, or a triumph for the party.\n"
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
        """
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.2,
                "stream": False, "id_slot": self.generate_slot, "reasoning_effort": "none"}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

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
        for d in Decision:
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
        else, and logs the raw answer, so the caller falls back to its keyword rules.
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
