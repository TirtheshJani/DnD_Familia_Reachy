import json

from familiar_brain.attention import Attention
from familiar_brain.state import BrainState

ON, OFF = ("head_tracking", {"enabled": True}), ("head_tracking", {"enabled": False})


def make():
    st = BrainState()
    return st, Attention(state=st)


def _calls(items):
    return [(i["name"], json.loads(i["arguments"])) for i in items]


def test_the_first_turn_switches_tracking_on_whoever_it_was_said_to():
    """state.tracking starts False, so the first line the brain hears, an IGNORE line
    included, turns the tracker on."""
    st, att = make()
    assert _calls(att.items()) == [ON]
    assert st.tracking is True


def test_tracking_stays_on_with_no_repeated_toggle():
    """A repeated head_tracking(true) resets the app's speaking handoff."""
    _, att = make()
    att.items()
    assert [att.items() for _ in range(5)] == [[]] * 5


def test_a_mute_switches_tracking_off_once_and_an_unmute_back_on():
    st, att = make()
    att.items()
    st.muted = True
    assert _calls(att.items()) == [OFF]
    assert att.items() == [] and st.tracking is False
    st.muted = False
    assert _calls(att.items()) == [ON]
    assert att.items() == []


def test_muted_from_the_start_sends_nothing():
    st, att = make()
    st.muted = True
    assert att.items() == [] and st.tracking is False
