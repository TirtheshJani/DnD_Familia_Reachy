import json
import math

from familiar_brain.attention import Attention
from familiar_brain.gate import Decision
from familiar_brain.state import BrainState


def make(doa=lambda: None, window=8.0, linger=4.0, glance_gap=12.0):
    st = BrainState()
    return st, Attention(state=st, doa=doa, follow_up_window_s=window, linger_s=linger, glance_gap_s=glance_gap)


def _calls(items):
    return [(i["name"], json.loads(i["arguments"])) for i in items]


def test_tracking_turns_on_when_addressed_and_off_after_the_window_plus_linger():
    st, att = make()
    st.mark_named(100.0)
    assert _calls(att.items(Decision.ADDRESSED, 100.0)) == [("head_tracking", {"enabled": True})]
    assert att.items(Decision.ADDRESSED, 103.0) == []                      # no change, no repeat
    assert att.items(Decision.IGNORE, 110.0) == []                         # 8 + 4 = 12 s of linger
    assert _calls(att.items(Decision.IGNORE, 112.5)) == [("head_tracking", {"enabled": False})]
    assert st.tracking is False


def test_glances_only_while_tracking_is_off_and_rate_limited():
    st, att = make(doa=lambda: (0.1, True))
    assert _calls(att.items(Decision.IGNORE, 0.0)) == [("move_head", {"direction": "left"})]
    assert att.items(Decision.AMBIENT, 5.0) == []                          # inside glance gap
    assert _calls(att.items(Decision.AMBIENT, 13.0)) == [("move_head", {"direction": "left"})]
    st.mark_named(20.0)
    assert _calls(att.items(Decision.ADDRESSED, 20.0)) == [("head_tracking", {"enabled": True})]
    assert att.items(Decision.IGNORE, 21.0) == []                          # tracking owns the head


def test_no_doa_means_no_glance():
    _, att = make(doa=lambda: None)
    assert att.items(Decision.IGNORE, 0.0) == []


def test_muted_never_tracks_and_never_glances():
    st, att = make(doa=lambda: (math.pi, True))
    st.mark_named(0.0); att.items(Decision.ADDRESSED, 0.0)
    st.muted = True
    assert _calls(att.items(Decision.IGNORE, 1.0)) == [("head_tracking", {"enabled": False})]
    assert att.items(Decision.IGNORE, 30.0) == []
    assert att.items(Decision.IGNORE, 60.0) == []


def test_linger_keeps_tracking_on_but_never_switches_it_on():
    """Mute while tracking, one muted turn switches it off, unmute: a line not spoken to
    Rocky inside the old linger must not switch tracking back on."""
    st, att = make()
    st.mark_named(0.0); att.items(Decision.ADDRESSED, 0.0)
    st.muted = True
    assert _calls(att.items(Decision.IGNORE, 2.0)) == [("head_tracking", {"enabled": False})]
    st.muted = False
    assert att.items(Decision.IGNORE, 9.0) == []
