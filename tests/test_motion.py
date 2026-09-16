import json
from pathlib import Path

from familiar_brain.motion import (
    REACTION_MOODS, Choreographer, table_mood, addressed_mood, tracking_item, glance_item,
)
from familiar_brain.persona import load_persona

PERSONA = """
name: Rocky
aliases: [rocky]
tts_speaker: Aiden
manner: dry
gestures:
  attentive: play_emotion:attentive
  amused: play_emotion:happy
  alarmed: play_emotion:surprised
  thoughtful: play_emotion:thinking
  sad: play_emotion:sad
  scared: play_emotion:scared
  asleep: play_emotion:sleepy
  awake: play_emotion:welcoming
"""


def persona(tmp_path: Path):
    p = tmp_path / "p.yaml"; p.write_text(PERSONA)
    return load_persona(p)


def test_table_mood_rules():
    assert table_mood("Natural twenty!") == "amused"
    assert table_mood("oh no, nat 1") == "alarmed"
    assert table_mood("He went down! Get him up!") == "scared"
    assert table_mood("hahaha that is hilarious") == "amused"
    assert table_mood("They hold a small funeral at dawn.") == "sad"
    assert table_mood("the door creaks") is None


def test_reaction_moods_are_the_four_narration_reactions():
    assert REACTION_MOODS == ("amused", "alarmed", "sad", "scared")


def test_addressed_mood_rules_default_to_attentive():
    assert addressed_mood("Rocky, roll me a d20") == "attentive"
    assert addressed_mood("Rocky, what do you think of him?") == "thoughtful"
    assert addressed_mood("good boy Rocky") == "proud"
    assert addressed_mood("Rocky, hello") == "attentive"


def test_tracking_and_glance_items():
    t = tracking_item(True)
    assert t["name"] == "head_tracking" and json.loads(t["arguments"]) == {"enabled": True}
    g = glance_item("left")
    assert g["name"] == "move_head" and json.loads(g["arguments"]) == {"direction": "left"}


def test_choreographer_uses_classifier_then_rules_and_rate_limits(tmp_path: Path):
    now = {"t": 100.0}
    calls = []
    def classify(text):
        calls.append(text); return "thoughtful"
    c = Choreographer(persona(tmp_path), min_gap_s=6.0, reaction_gap_s=0.0,
                      clock=lambda: now["t"], mood_classifier=classify)
    items = c.for_addressed("Rocky, roll a d20", now["t"])
    assert [json.loads(i["arguments"])["emotion"] for i in items] == ["thinking"]
    assert calls == ["Rocky, roll a d20"]
    now["t"] += 2
    assert c.for_addressed("Rocky, again", now["t"]) == []          # inside the gap
    now["t"] += 5
    assert len(c.for_addressed("Rocky, again", now["t"])) == 1       # gap elapsed


def test_choreographer_falls_back_when_classifier_fails_or_answers_nonsense(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                      mood_classifier=lambda t: (_ for _ in ()).throw(RuntimeError("down")))
    assert json.loads(c.for_addressed("Rocky, what do you think", 0.0)[0]["arguments"]) == {"emotion": "thinking"}
    c2 = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                       mood_classifier=lambda t: "banana")
    assert json.loads(c2.for_addressed("Rocky, hello", 0.0)[0]["arguments"]) == {"emotion": "attentive"}


def test_table_reaction_uses_rules_first_and_cues_ignore_the_gap(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=60.0, reaction_gap_s=0.0, clock=lambda: 0.0)
    assert c.for_table("the door creaks", 0.0) == []
    assert json.loads(c.for_table("natural 20", 0.0)[0]["arguments"]) == {"emotion": "happy"}
    assert c.for_table("natural 20", 1.0) == []                       # rate limited
    assert json.loads(c.cue("asleep", 1.0)[0]["arguments"]) == {"emotion": "sleepy"}   # cues always fire


def test_table_reaction_asks_the_classifier_for_narration_of_five_words_or_more(tmp_path: Path):
    asked = []
    def react(text):
        asked.append(text); return "sad"
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=15.0, clock=lambda: 0.0,
                      reaction_classifier=react)
    line = "The letter is from her brother, dead these ten years."
    assert json.loads(c.for_table(line, 0.0)[0]["arguments"]) == {"emotion": "sad"}
    assert asked == [line]
    assert c.for_table("A shape moves behind the altar and laughs.", 5.0) == []   # inside reaction gap, no call
    assert asked == [line]
    assert c.for_table("short line here", 20.0) == [] and asked == [line]        # under five words, no call


def test_table_reaction_treats_none_nonsense_and_errors_as_no_gesture(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                      reaction_classifier=lambda t: "none")
    assert c.for_table("You walk for an hour along the road.", 0.0) == []
    c2 = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                       reaction_classifier=lambda t: "thoughtful")               # not a reaction mood
    assert c2.for_table("You walk for an hour along the road.", 0.0) == []
    c3 = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                       reaction_classifier=lambda t: (_ for _ in ()).throw(RuntimeError("down")))
    assert c3.for_table("You walk for an hour along the road.", 0.0) == []


def test_mood_without_a_gesture_yields_nothing(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0)
    assert c.cue("proud", 0.0) == []   # persona above defines no proud gesture
