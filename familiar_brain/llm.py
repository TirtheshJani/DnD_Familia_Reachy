from __future__ import annotations

from typing import Sequence

import requests

from familiar_brain.gate import Decision

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


class LLMClient:
    def __init__(self, *, base_url: str, classify_slot: int, classify_timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.classify_slot = classify_slot
        self.classify_timeout_s = classify_timeout_s

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
