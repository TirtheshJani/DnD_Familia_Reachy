import logging

import pytest

from familiar_brain.modes import FAMILIAR, THINKING, update_mode
from familiar_brain.state import BrainState


def test_enter_and_exit_phrases_on_addressed_turns_only():
    st = BrainState()
    assert update_mode(st, "Rocky, think with me about the cult", addressed=True, now=0.0, timeout_s=300) == THINKING
    assert update_mode(st, "let's think about lunch", addressed=False, now=1.0, timeout_s=300) == THINKING
    assert update_mode(st, "Rocky, what about the docks?", addressed=True, now=2.0, timeout_s=300) == THINKING
    assert update_mode(st, "Rocky, thanks, that's all", addressed=True, now=3.0, timeout_s=300) == FAMILIAR


def test_thinking_times_out_without_addressed_turns():
    st = BrainState()
    update_mode(st, "Rocky, help me think this through", addressed=True, now=0.0, timeout_s=300)
    update_mode(st, "Rocky, and?", addressed=True, now=100.0, timeout_s=300)   # keeps it alive
    assert update_mode(st, "table talk", addressed=False, now=399.0, timeout_s=300) == THINKING
    assert update_mode(st, "table talk", addressed=False, now=401.0, timeout_s=300) == FAMILIAR


def test_exit_phrases_win_over_enter_phrases_in_one_line():
    st = BrainState()
    assert update_mode(st, "Rocky, quick answer, no need to talk it through", addressed=True, now=0.0, timeout_s=300) == FAMILIAR


def _thinking() -> BrainState:
    st = BrainState()
    assert update_mode(st, "Rocky, think with me", addressed=True, now=0.0, timeout_s=300) == THINKING
    return st


@pytest.mark.parametrize("line", [
    "Rocky, what are my options for a bonus action?",
    "Rocky, let's think about lunch",
])
def test_ordinary_table_questions_do_not_open_thinking(line):
    assert update_mode(BrainState(), line, addressed=True, now=0.0, timeout_s=300) == FAMILIAR


@pytest.mark.parametrize("line", [
    "Rocky, help me think",
    "Rocky, think this through with me",
    "Rocky, let's talk it through",
    "Rocky, be my thought partner",
])
def test_d8_phrases_open_thinking(line):
    assert update_mode(BrainState(), line, addressed=True, now=0.0, timeout_s=300) == THINKING


@pytest.mark.parametrize("line", [
    "Rocky, thanks for that",
    "Rocky, that's it!",
    "Rocky, that's enough gold?",
    "Rocky, that's all the gold?",
    "Rocky, that's all?",
    "Rocky, that's all for now?",
    "Rocky, that's all...?",
])
def test_table_talk_does_not_close_thinking(line):
    st = _thinking()
    assert update_mode(st, line, addressed=True, now=1.0, timeout_s=300) == THINKING


@pytest.mark.parametrize("line", [
    "Rocky, that\u2019s all",
    "That's all for now, Rocky.",
    "Rocky, thats all.",
    "Rocky, that is all!",
    "Rocky, quick answer: can I dash?",
])
def test_closing_phrases_close_thinking(line):
    st = _thinking()
    assert update_mode(st, line, addressed=True, now=1.0, timeout_s=300) == FAMILIAR


def test_the_timeout_runs_from_the_last_addressed_turn_not_from_opening():
    st = _thinking()
    update_mode(st, "Rocky, and the docks?", addressed=True, now=250.0, timeout_s=300)
    assert update_mode(st, "table talk", addressed=False, now=500.0, timeout_s=300) == THINKING
    # Table talk never refreshed the clock, so an addressed turn past the timeout finds
    # the familiar already back to two sentences and does not revive thinking mode.
    assert update_mode(st, "Rocky, and the ship?", addressed=True, now=551.0, timeout_s=300) == FAMILIAR


def test_the_phrase_that_changed_the_mode_is_logged(caplog):
    caplog.set_level(logging.INFO, logger="familiar.modes")
    st = BrainState()
    update_mode(st, "Rocky, be my thought partner", addressed=True, now=0.0, timeout_s=300)
    update_mode(st, "Rocky, quick answer: can I dash?", addressed=True, now=1.0, timeout_s=300)
    assert "thought partner" in caplog.text and "quick answer" in caplog.text
