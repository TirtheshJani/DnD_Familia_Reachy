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
