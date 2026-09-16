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

# Both lists run on every addressed line, so they hold only the phrases D8 names and a
# few nobody says in ordinary table talk. Broader ones misfired: "what are my options
# for a bonus action?" opened thinking mode, and "thanks for that" or "that's all the
# gold?" closed it. "That's all" closes only as a sign-off, at the end of the line or
# before punctuation that does not end in a question mark. U+2019 is accepted wherever
# an apostrophe can appear, since transcripts use either. If these still misfire, D8's
# reversal applies: keep only POST /api/mode.
ENTER = re.compile(
    r"\b(think ((it|this|that) through )?with me|help me think|talk (it|this|that) through"
    r"|thought partner)\b", re.I)
EXIT = re.compile(
    r"\b(quick answer|short answer|back to normal|stop thinking)\b"
    r"|\bthat(['\u2019]?s| is) all( for now)?(?=\s*($|[.!,;:]+(?![.!,;:?])))", re.I)


def update_mode(state: BrainState, text: str, *, addressed: bool, now: float, timeout_s: float) -> str:
    if (state.mode == THINKING and state.mode_set_at is not None
            and now - state.mode_set_at > timeout_s):
        log.info("thinking mode timed out")
        state.mode = FAMILIAR
    if not addressed:
        return state.mode
    # The matched phrase goes in the log so a misfire at the table can be traced to it.
    if closing := EXIT.search(text):
        if state.mode != FAMILIAR:
            log.info("thinking mode closed by phrase %r", closing.group(0))
        state.mode = FAMILIAR
    elif opening := ENTER.search(text):
        if state.mode != THINKING:
            log.info("thinking mode opened by phrase %r", opening.group(0))
        state.mode = THINKING
    if state.mode == THINKING:
        state.mode_set_at = now
    return state.mode
