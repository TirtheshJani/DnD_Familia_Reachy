"""Reply modes: the two-sentence familiar, or a thinking partner when asked.

The mode changes one fixed block of the system prompt (persona.LENGTH_BLOCKS) and
nothing else. It is opened and closed by phrases on ADDRESSED turns, by the companion
UI through POST /api/mode, and it expires after mode_timeout_s without an addressed
turn so a table never inherits a chatty familiar from an hour ago.
"""
from __future__ import annotations

import logging
import re

from familiar_brain.state import BrainState

log = logging.getLogger("familiar.modes")

FAMILIAR = "familiar"
THINKING = "thinking"
MODES = (FAMILIAR, THINKING)

ENTER = re.compile(
    r"\b(think (it |this |that )?(through )?with me|let'?s think|help me think|thought partner"
    r"|talk (it|this|that) through|what are (my|our) options)\b", re.I)
EXIT = re.compile(
    r"\b(that'?s (all|enough|it)|thank you|thanks|quick answer|short answer|back to normal"
    r"|stop thinking)\b", re.I)


def update_mode(state: BrainState, text: str, *, addressed: bool, now: float, timeout_s: float) -> str:
    if (state.mode == THINKING and state.mode_set_at is not None
            and now - state.mode_set_at > timeout_s):
        log.info("thinking mode timed out")
        state.mode = FAMILIAR
    if not addressed:
        return state.mode
    if EXIT.search(text):
        if state.mode != FAMILIAR:
            log.info("thinking mode closed by phrase")
        state.mode = FAMILIAR
    elif ENTER.search(text):
        if state.mode != THINKING:
            log.info("thinking mode opened by phrase")
        state.mode = THINKING
    if state.mode == THINKING:
        state.mode_set_at = now
    return state.mode
