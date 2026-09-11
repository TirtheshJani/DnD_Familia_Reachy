from familiar_brain.state import BrainState
from familiar_brain.gate import Gate, Decision


def make(now=1000.0, window=8.0):
    st = BrainState()
    g = Gate(state=st, name="Understudy", aliases=("understudy", "little one"),
             follow_up_window_s=window, classifier=None, clock=lambda: now)
    return st, g


def test_muted_returns_ignore_even_when_named():
    st, g = make()
    st.muted = True
    assert g.decide(["Understudy, roll for me"]) == Decision.IGNORE


def test_name_in_segment_is_addressed():
    _, g = make()
    assert g.decide(["Hey Understudy, what is a grapple?"]) == Decision.ADDRESSED


def test_alias_word_boundary_only():
    _, g = make()
    assert g.decide(["the little one hides"]) == Decision.ADDRESSED
    assert g.decide(["I misunderstudy nothing"]) == Decision.IGNORE


def test_speak_now_arms_one_turn():
    st, g = make()
    st.arm_speak_now()
    assert g.decide(["what do you think"]) == Decision.ADDRESSED
    assert g.decide(["what do you think"]) == Decision.IGNORE


def test_follow_up_window_open_then_closed():
    st = BrainState()
    now = {"t": 1000.0}
    g = Gate(state=st, name="U", aliases=(), follow_up_window_s=8.0,
             classifier=None, clock=lambda: now["t"])
    st.mark_spoke(at=1000.0)
    now["t"] = 1005.0
    assert g.decide(["and then?"]) == Decision.ADDRESSED
    now["t"] = 1009.0
    assert g.decide(["and then?"]) == Decision.IGNORE


def test_follow_up_window_does_not_chain_forever():
    """Layer 2 is anchored to the last time the familiar spoke, and speaking re-arms it.
    At a table where someone talks every few seconds that window never closes, so the
    familiar answers every utterance for the rest of the session. Observed live: six
    consecutive ADDRESSED decisions at gate_ms 0.0, none of them addressed to it.
    The chain is capped so it always returns to the real gate."""
    st = BrainState()
    now = {"t": 1000.0}
    seen = []

    def clf(segments):
        seen.append(segments[-1])
        return Decision.IGNORE

    g = Gate(state=st, name="Understudy", aliases=(), follow_up_window_s=8.0,
             classifier=clf, clock=lambda: now["t"], max_follow_ups=3)
    assert g.decide(["Understudy, what do you see?"]) == Decision.ADDRESSED
    for i in range(3):  # the granted follow-ups
        st.mark_spoke(at=now["t"])  # speaking re-arms the window
        now["t"] += 1.0
        assert g.decide([f"line {i}"]) == Decision.ADDRESSED, f"follow-up {i} should be granted"
    st.mark_spoke(at=now["t"])
    now["t"] += 1.0
    assert g.decide(["still chatting"]) == Decision.IGNORE, "the chain must end"
    assert seen == ["still chatting"], "the classifier must get the say once the cap is hit"


def test_explicit_address_resets_the_follow_up_chain():
    st = BrainState()
    now = {"t": 1000.0}
    g = Gate(state=st, name="Understudy", aliases=(), follow_up_window_s=8.0,
             classifier=lambda s: Decision.IGNORE, clock=lambda: now["t"], max_follow_ups=2)
    g.decide(["Understudy, hello"])
    for i in range(2):
        st.mark_spoke(at=now["t"]); now["t"] += 1.0
        assert g.decide([f"f{i}"]) == Decision.ADDRESSED
    st.mark_spoke(at=now["t"]); now["t"] += 1.0
    assert g.decide(["exhausted"]) == Decision.IGNORE
    # naming it again starts a fresh conversation, and fresh follow-ups
    assert g.decide(["Understudy, again"]) == Decision.ADDRESSED
    st.mark_spoke(at=now["t"]); now["t"] += 1.0
    assert g.decide(["and now?"]) == Decision.ADDRESSED


def test_classifier_used_only_when_earlier_layers_say_no():
    calls = []
    def clf(segments):
        calls.append(segments); return Decision.AMBIENT
    st = BrainState()
    g = Gate(state=st, name="U", aliases=(), follow_up_window_s=8.0, classifier=clf, clock=lambda: 0.0)
    assert g.decide(["U, hello"]) == Decision.ADDRESSED and calls == []
    assert g.decide(["a natural twenty!"]) == Decision.AMBIENT and len(calls) == 1


def test_classifier_exception_defaults_to_ignore():
    def clf(_): raise TimeoutError
    st = BrainState()
    g = Gate(state=st, name="U", aliases=(), follow_up_window_s=8.0, classifier=clf, clock=lambda: 0.0)
    assert g.decide(["mumble"]) == Decision.IGNORE
