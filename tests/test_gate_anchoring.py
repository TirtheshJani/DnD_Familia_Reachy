from familiar_brain.gate import Decision, Gate
from familiar_brain.state import BrainState


def make(clock):
    st = BrainState()
    g = Gate(state=st, name="Understudy", aliases=("little one",), follow_up_window_s=8.0,
             classifier=lambda segs: Decision.IGNORE, clock=clock)
    return st, g


def test_reply_to_reply_does_not_re_arm_window():
    """Reproduces the Plan 1 failure: six consecutive grants with nobody naming it.

    Every mark_spoke passes an explicit `at`. Bare mark_spoke() stamps time.time(),
    which no fake clock can reach, so the extension branch would never be entered
    and the test would pass for the wrong reason.
    """
    now = {"t": 1000.0}
    st, g = make(lambda: now["t"])
    assert g.decide(["Understudy, hello"]) == Decision.ADDRESSED
    st.mark_spoke(at=now["t"])             # familiar replies at t=1000
    now["t"] += 3
    assert g.decide(["and then?"]) == Decision.ADDRESSED   # one follow-up is fine
    st.mark_spoke(at=now["t"])             # reply at t=1003
    now["t"] += 3                          # t=1006, still inside the single extension
    assert g.decide(["okay moving on"]) == Decision.ADDRESSED
    st.mark_spoke(at=now["t"])
    now["t"] += 3
    # window = last_named_at + 8 = 1008, extended once from the FIRST reply to
    # 1000 + 8 = 1008. The reply at 1003 does not push it out. t=1009 is outside.
    assert g.decide(["someone else talking"]) == Decision.IGNORE
    st.mark_spoke(at=now["t"])             # even if it somehow spoke, no re-arm
    now["t"] += 1
    assert g.decide(["more table talk"]) == Decision.IGNORE


def test_one_off_extension_holds_until_named_again():
    """The first reply pushes the deadline from 0 + 8 to 3 + 8 = 11. That has to hold on
    every later turn, not only the one that took the extension: at t=10 the unextended
    deadline of 8 has passed but the extended one has not."""
    now = {"t": 0.0}
    seen = []

    def clf(segs):
        seen.append(segs[-1])
        return Decision.IGNORE

    st = BrainState()
    g = Gate(state=st, name="Understudy", aliases=(), follow_up_window_s=8.0,
             classifier=clf, clock=lambda: now["t"], max_follow_ups=3)
    assert g.decide(["Understudy, which way?"]) == Decision.ADDRESSED
    st.mark_spoke(at=3.0)
    now["t"] = 6.0
    assert g.decide(["and after that?"]) == Decision.ADDRESSED
    now["t"] = 10.0
    assert g.decide(["are you sure?"]) == Decision.ADDRESSED
    now["t"] = 11.5
    assert g.decide(["right, Gustave, your turn"]) == Decision.IGNORE
    assert seen == ["right, Gustave, your turn"], "past the extended deadline the classifier decides"


def test_naming_again_resets_the_extension():
    now = {"t": 0.0}
    st, g = make(lambda: now["t"])
    g.decide(["Understudy?"])
    st.mark_spoke(at=3.0)
    now["t"] = 6.0
    assert g.decide(["go on"]) == Decision.ADDRESSED    # extended to 11
    now["t"] = 50.0
    assert g.decide(["Understudy, again"]) == Decision.ADDRESSED
    now["t"] = 57.0
    assert g.decide(["and?"]) == Decision.ADDRESSED     # fresh window, 50 + 8
    now["t"] = 58.5
    assert g.decide(["table talk"]) == Decision.IGNORE, "the old extension must not carry over"


def test_naming_again_reopens_window():
    now = {"t": 0.0}
    st, g = make(lambda: now["t"])
    g.decide(["Understudy?"]); st.mark_spoke(at=now["t"])
    now["t"] = 100.0
    assert g.decide(["what now"]) == Decision.IGNORE
    assert g.decide(["little one, again"]) == Decision.ADDRESSED
    now["t"] = 104.0
    assert g.decide(["and?"]) == Decision.ADDRESSED


def test_speak_now_counts_as_naming():
    now = {"t": 0.0}
    st, g = make(lambda: now["t"])
    st.arm_speak_now()
    assert g.decide(["what do you think"]) == Decision.ADDRESSED
    now["t"] = 4.0
    assert g.decide(["really?"]) == Decision.ADDRESSED


def test_max_follow_ups_guard_still_applies():
    now = {"t": 0.0}
    st, g = make(lambda: now["t"])
    g.decide(["Understudy"])
    for i in range(3):
        now["t"] += 0.5
        assert g.decide([f"follow {i}"]) == Decision.ADDRESSED
    now["t"] += 0.5
    assert g.decide(["fourth"]) == Decision.IGNORE
