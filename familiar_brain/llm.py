from __future__ import annotations

from typing import Sequence

import requests

from familiar_brain.gate import Decision

CLASSIFY_INSTRUCTION = (
    "You watch a tabletop game. A creature named {name} sits at the table. Read the recent lines "
    "(most recent last). Decide about the LAST line only. Answer with exactly one word:\n"
    "ADDRESSED if the last line is spoken to {name} or asks {name} something.\n"
    "AMBIENT if it is exciting or emotional table talk {name} might react to silently.\n"
    "IGNORE otherwise."
)


class LLMClient:
    def __init__(self, *, base_url: str, classify_slot: int, classify_timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.classify_slot = classify_slot
        self.classify_timeout_s = classify_timeout_s

    def classify(self, segments: Sequence[str], *, persona_name: str) -> Decision:
        lines = "\n".join(f"- {s}" for s in segments[-4:])
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
