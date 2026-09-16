import pytest

from familiar_brain.state import BrainState
from familiar_brain.gate import Gate, Decision


def make(now=1000.0, window=8.0):
    """Returns the clock as a mutable dict: the follow-up window now hangs off the
    moment the familiar was named, so a test that names it and then expects IGNORE
    has to let the clock run past the window."""
    clock = {"t": now}
    st = BrainState()
    g = Gate(state=st, name="Understudy", aliases=("understudy", "little one"),
             follow_up_window_s=window, classifier=None, clock=lambda: clock["t"])
    return st, g, clock


def test_muted_returns_ignore_even_when_named():
    st, g, _ = make()
    st.muted = True
    assert g.decide(["Understudy, roll for me"]) == Decision.IGNORE


def test_name_in_segment_is_addressed():
    _, g, _ = make()
    assert g.decide(["Hey Understudy, what is a grapple?"]) == Decision.ADDRESSED


def test_alias_word_boundary_only():
    _, g, clock = make()
    assert g.decide(["the little one hides"]) == Decision.ADDRESSED
    clock["t"] += 20  # the alias opened the follow-up window; let it close
    assert g.decide(["I misunderstudy nothing"]) == Decision.IGNORE


def test_speak_now_arms_one_turn():
    st, g, clock = make()
    st.arm_speak_now()
    assert g.decide(["what do you think"]) == Decision.ADDRESSED
    clock["t"] += 20  # Speak Now counts as naming, so let its window close too
    assert g.decide(["what do you think"]) == Decision.IGNORE


def test_follow_up_window_open_then_closed():
    st = BrainState()
    now = {"t": 1000.0}
    g = Gate(state=st, name="U", aliases=(), follow_up_window_s=8.0,
             classifier=None, clock=lambda: now["t"])
    assert g.decide(["U, are you there?"]) == Decision.ADDRESSED  # naming opens the window
    st.mark_spoke(at=1000.0)
    now["t"] = 1005.0
    assert g.decide(["and then?"]) == Decision.ADDRESSED
    now["t"] = 1009.0
    assert g.decide(["and then?"]) == Decision.IGNORE


def test_follow_up_window_does_not_chain_forever():
    """Observed live: six consecutive ADDRESSED decisions at gate_ms 0.0, none of them
    addressed to it. Two things now stop that chain, and this test pins the cap. The
    window is anchored to naming (see tests/test_gate_anchoring.py), and on top of that
    only max_follow_ups grants are allowed before the gate hands back to the classifier,
    so even a rapid exchange inside one open window terminates."""
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
        st.mark_spoke(at=now["t"])
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
    now = {"t": 0.0}
    g = Gate(state=st, name="U", aliases=(), follow_up_window_s=8.0, classifier=clf, clock=lambda: now["t"])
    assert g.decide(["U, hello"]) == Decision.ADDRESSED and calls == []
    now["t"] += 20  # let the naming window close so layer 3 is the one that answers
    assert g.decide(["a natural twenty!"]) == Decision.AMBIENT and len(calls) == 1


def test_classifier_exception_defaults_to_ignore():
    def clf(_): raise TimeoutError
    st = BrainState()
    g = Gate(state=st, name="U", aliases=(), follow_up_window_s=8.0, classifier=clf, clock=lambda: 0.0)
    assert g.decide(["mumble"]) == Decision.IGNORE


def rocky():
    """The live persona: config/persona.yaml names Rocky with the alias rocky, and the
    lexicon has already turned Rockie, Rockey, Roki and Rocki into Rocky."""
    seen = []

    def clf(segments):
        seen.append(segments[-1])
        return Decision.IGNORE

    st = BrainState()
    g = Gate(state=st, name="Rocky", aliases=("rocky",), follow_up_window_s=8.0,
             classifier=clf, clock=lambda: 0.0)
    return st, g, seen


@pytest.mark.parametrize("line", [
    "The path ahead is rocky.",
    "You climb a rocky outcrop.",
    "the rocky shore stretches north",
    "It's very rocky here",
    "The ground is rocky, so watch your step",
    "Rocky terrain slows you down",
    "a rocky start to the negotiations",
])
def test_rocky_the_adjective_is_not_a_naming(line):
    st, g, seen = rocky()
    assert g.decide([line]) == Decision.IGNORE
    assert seen == [line], "an adjective use must fall through to the classifier"
    assert st.last_named_at is None


@pytest.mark.parametrize("line", [
    "Rocky, roll a d20",
    "Hey Rocky",
    "thanks Rocky",
    "What do you think, Rocky?",
    "Is Rocky awake?",
    "Where is Rocky?",
    "Rocky what do you see",
    "ROCKY!",
    "Rocky's turn to guess",
    "good boy Rocky",
    "Rocky, is the path rocky?",
])
def test_rocky_spoken_to_or_about_is_a_naming(line):
    st, g, seen = rocky()
    assert g.decide([line]) == Decision.ADDRESSED
    assert seen == [], "a naming never reaches the classifier"
    assert st.last_named_at == 0.0


def test_adjective_use_falls_through_to_an_open_window_without_renaming():
    """Inside an open window the line is still a follow-up, but it must not count as
    naming: that would reset the follow-up cap and reopen the window from now."""
    now = {"t": 0.0}
    st = BrainState()
    g = Gate(state=st, name="Rocky", aliases=("rocky",), follow_up_window_s=8.0,
             classifier=lambda s: Decision.IGNORE, clock=lambda: now["t"])
    assert g.decide(["Rocky, which way?"]) == Decision.ADDRESSED
    now["t"] = 7.0
    assert g.decide(["The path ahead is rocky."]) == Decision.ADDRESSED
    assert st.last_named_at == 0.0 and st.follow_ups == 1


def test_adjective_guard_applies_to_any_persona_name():
    st = BrainState()
    g = Gate(state=st, name="Sunny", aliases=(), follow_up_window_s=8.0,
             classifier=lambda s: Decision.IGNORE, clock=lambda: 0.0)
    assert g.decide(["The morning is sunny."]) == Decision.IGNORE
    assert g.decide(["Sunny, what now?"]) == Decision.ADDRESSED
