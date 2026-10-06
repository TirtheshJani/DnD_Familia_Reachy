"""Rocky replies only when spoken to.

Live, 2026-09-16 (data/sessions/20260916-182232-0189cc.jsonl): of about seventy lines,
the one time Rocky spoke uninvited was an 868 word segment in which the transcriber had
merged minutes of table chatter. The players were talking ABOUT the familiar in the
middle of it ("we're doing uh rocky", "What's his name, sorry? Rocky."), the gate found
the name, and Rocky generated for a minute. That row is kept verbatim in
tests/fixtures/live_uninvited_rocky.jsonl, since data/ is not tracked.
"""
import json
from pathlib import Path

import pytest

from familiar_brain.gate import Decision, Gate
from familiar_brain.state import BrainState

LIVE = json.loads((Path(__file__).parent / "fixtures" / "live_uninvited_rocky.jsonl")
                  .read_text(encoding="utf-8"))


def rocky(classifier=None, now=None):
    """The live persona, with a classifier that records what reaches it."""
    seen = []
    clock = now if now is not None else {"t": 0.0}

    def clf(segments):
        seen.append(segments[-1])
        return Decision.IGNORE

    st = BrainState()
    g = Gate(state=st, name="Rocky", aliases=("rocky",), follow_up_window_s=8.0,
             classifier=classifier or clf, clock=lambda: clock["t"])
    return st, g, seen


def test_the_live_run_on_segment_is_not_a_naming():
    assert LIVE["spoke"] is True and LIVE["decision"] == "ADDRESSED", "the row the fix is for"
    st, g, _ = rocky()
    assert g.decide([LIVE["text"]]) == Decision.IGNORE
    assert st.last_named_at is None


def test_a_naming_buried_early_in_a_run_on_segment_does_not_count():
    chatter = " ".join(["and then we went back to the car park"] * 7)
    assert len(chatter.split()) > 60
    st, g, _ = rocky()
    assert g.decide([f"Rocky, what's the DC? {chatter}"]) == Decision.IGNORE
    assert st.last_named_at is None


@pytest.mark.parametrize("line", [
    "Rocky, can you remind us what happened last session when we went into the tower and "
    "fought the wizard and then what did the Gatewarden say to us about the keys and the gate",
    "Rocky what do you think we should do we could go left to the tunnels or right to the "
    "temple or back to the inn and rest up before the fight with the dragon tonight",
])
def test_a_long_request_that_starts_with_the_name_still_counts(line):
    """The 25 word limit is for run-ons the transcriber glued together. A segment of 60
    words or fewer is one utterance, and a recap request to the familiar runs this long."""
    assert 25 < len(line.split()) <= 60
    _, g, _ = rocky()
    assert g.decide([line]) == Decision.ADDRESSED


def test_a_naming_at_the_end_of_a_long_segment_still_counts():
    """The transcriber merges chatter in front of the line the player has just said to
    Rocky, so a naming near the end is still the line that was spoken to it."""
    chatter = " ".join(["and then we went back to the car park"] * 10)
    st, g, _ = rocky()
    assert g.decide([f"{chatter} okay Rocky what do you think we should do"]) == Decision.ADDRESSED


def test_a_long_unpunctuated_request_with_the_name_first_still_counts():
    line = ("Rocky can you remind us what the old man at the gate said about the "
            "tunnels under the temple and the bells")
    _, g, _ = rocky()
    assert g.decide([line]) == Decision.ADDRESSED


@pytest.mark.parametrize("line", [
    "his name is Rocky and he's from a movie",
    "What's his name? Rocky.",
    "What's his name, sorry? Rocky. Originally from the Project Hail Mary movie.",
    "it's called Rocky",
    "a little guy named Rocky",
    "I call him Rocky because of the film",
    "we're doing rocky",
    "Um, we're doing uh rocky. Okay, understood.",
    "we were talking about Rocky earlier",
    "Rocky's eyes are glowing",
    "we named him Rocky",
    "he named it rocky",
    "What do you call him? Rocky.",
    "look, Rocky\u2019s ears moved",
])
def test_talking_about_rocky_is_not_a_naming(line):
    st, g, seen = rocky()
    assert g.decide([line]) == Decision.IGNORE
    assert st.last_named_at is None
    assert seen == [line], "a mention falls through to the follow-up window and classifier"


@pytest.mark.parametrize("line", [
    "Rocky, what's the DC?",
    "Hey Rocky pass me the dice",
    "Me too Rocky",
    "is Rocky awake?",
    "thanks Rocky",
    "Rocky what do you see",
    "what do you think, Rocky?",
    "Rocky?",
    "what are you doing Rocky",
    "Rocky's turn to guess",
    "Rocky, his name is Bob",
    # Recall questions. The name answers no question about its own name, and with the
    # comma dropped "called", "named" or "talking about" ends a question put to Rocky.
    "what was the innkeeper's name? Rocky?",
    "What's the blacksmith's name? Rocky, do you remember?",
    "So what's his name? Rocky, do you know?",
    "what's this called Rocky",
    "what's that spell called Rocky",
    "what was the tavern called Rocky",
    "what is this place called rocky",
    "what was the dragon named Rocky",
    "what was that guy named rocky",
    "what were we talking about Rocky",
    "what did we talk about Rocky",
    "what should I ask about Rocky",
    "who did we ask about rocky",
])
def test_speaking_to_rocky_is_still_a_naming(line):
    st, g, seen = rocky()
    assert g.decide([line]) == Decision.ADDRESSED
    assert seen == [] and st.last_named_at == 0.0


@pytest.mark.parametrize("line", ["Rocky is so cute", "I think Rocky is broken", "look at Rocky"])
def test_a_short_statement_about_rocky_still_reads_as_a_naming(line):
    """The limit of gating on the name: a short line about the familiar has nothing in its
    words to tell it from one said to it. Pinned so that nobody reads P13 as fixing it."""
    _, g, _ = rocky()
    assert g.decide([line]) == Decision.ADDRESSED


def test_the_classifier_can_no_longer_turn_an_unnamed_line_into_speech(caplog):
    """It used to answer ADDRESSED for "a direct question that only Rocky could answer".
    Now speech needs a naming, the window after one, or Speak Now."""
    st, g, _ = rocky(classifier=lambda segs: Decision.ADDRESSED)
    with caplog.at_level("INFO", logger="familiar.gate"):
        assert g.decide(["what's behind the door?"]) == Decision.IGNORE
    assert "ADDRESSED" in caplog.text
    assert st.last_named_at is None


def test_the_classifier_still_decides_ambient():
    _, g, _ = rocky(classifier=lambda segs: Decision.AMBIENT)
    assert g.decide(["a natural twenty!"]) == Decision.AMBIENT


def test_a_run_on_segment_inside_the_window_is_not_a_follow_up():
    now = {"t": 0.0}
    st, g, seen = rocky(now=now)
    assert g.decide(["Rocky, which way?"]) == Decision.ADDRESSED
    st.mark_spoke(at=1.0)
    now["t"] = 4.0
    chatter = " ".join(["so we parked up and then the shop was shut"] * 7)
    assert len(chatter.split()) > 60
    assert g.decide([chatter]) == Decision.IGNORE
    assert seen == [chatter], "table chatter goes to the classifier, for AMBIENT at most"
    assert st.follow_ups == 0, "chatter must not use up a follow-up"
    now["t"] = 5.0
    assert g.decide(["left or right?"]) == Decision.ADDRESSED, "a real follow-up still lands"


def test_a_long_follow_up_inside_the_window_still_counts():
    now = {"t": 0.0}
    st, g, _ = rocky(now=now)
    assert g.decide(["Rocky, what's down there?"]) == Decision.ADDRESSED
    now["t"] = 3.0
    line = " ".join(["and what about the bells"] * 10)  # 50 words
    assert g.decide([line]) == Decision.ADDRESSED
