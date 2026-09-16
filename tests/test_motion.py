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
    assert table_mood("critical fail") == "alarmed" and table_mood("crit fail") == "alarmed"
    assert table_mood("not a crit, a nat 1") == "alarmed" and table_mood("Critical hit!") == "amused"
    assert table_mood("I fumbled") == "alarmed"
    assert table_mood("hahahaha") == "amused" and table_mood("ahaha") == "amused" and table_mood("Aha! The key!") is None
    assert table_mood("they're down") == "scared" and table_mood("the drawbridge is down") is None


def test_reaction_moods_are_the_four_narration_reactions():
    assert REACTION_MOODS == ("amused", "alarmed", "sad", "scared")


def test_addressed_mood_rules_say_none_when_nothing_matches():
    """Like table_mood: None means no rule matched, so the caller can tell a roll, which
    the rules answer attentive, from a line the rules know nothing about."""
    assert addressed_mood("Rocky, roll me a d20") == "attentive"
    assert addressed_mood("Rocky, what do you think of him?") == "thoughtful"
    assert addressed_mood("good boy Rocky") == "proud"
    assert addressed_mood("Rocky, hello") is None


def test_addressed_rules_catch_a_warning_laughter_and_a_request_to_think():
    assert addressed_mood("Rocky, is that a dragon behind us?") == "alarmed"
    assert addressed_mood("Rocky, look out!") == "alarmed" and addressed_mood("watch out, Rocky") == "alarmed"
    assert addressed_mood("Rocky, look out") == "alarmed" and addressed_mood("Rocky, look out behind you!") == "alarmed"
    assert addressed_mood("Rocky, are we in danger?") == "alarmed"
    assert addressed_mood("Rocky, that is hilarious") == "amused" and addressed_mood("hahaha Rocky") == "amused"
    assert addressed_mood("Rocky, good joke") == "amused"
    assert addressed_mood("Rocky, help me think about the heist") == "thoughtful"
    assert addressed_mood("Rocky, think with me about the cult") == "thoughtful"


def test_addressed_rules_stay_out_of_ordinary_questions():
    """A rule hit wins over the mood call, so a rule that misfires is a wrong gesture the
    model could have avoided. Dragons are the game's name and its subject matter, not a
    threat on their own. Asking Rocky to keep watch is a plain request at a heist table,
    not a warning, and putting a quarrel behind us is not a threat behind us."""
    for line in ("Rocky, what do you know about dragons?", "Rocky, do you like Dungeons and Dragons?",
                 "Rocky, is this door dangerous?", "Rocky, what is behind the door?",
                 "Aha, Rocky, the key!", "Rocky, tell me a joke", "Rocky, poor Tomas is dead",
                 "Rocky, keep a look out for guards.", "Rocky, look out for traps in the next room.",
                 "Rocky, watch out for the city watch while we pick the lock.", "Rocky, keep a look out.",
                 "Rocky, look out the window.", "Rocky, lets put that behind us.",
                 "Rocky, we left the guards behind us.", "Rocky, you are too clever by half."):
        assert addressed_mood(line) is None, line


def test_tracking_and_glance_items():
    t = tracking_item(True)
    assert t["name"] == "head_tracking" and json.loads(t["arguments"]) == {"enabled": True}
    g = glance_item("left")
    assert g["name"] == "move_head" and json.loads(g["arguments"]) == {"direction": "left"}


def test_choreographer_asks_the_classifier_when_no_rule_matches_and_rate_limits(tmp_path: Path):
    now = {"t": 100.0}
    calls = []
    def classify(text):
        calls.append(text); return "thoughtful"
    c = Choreographer(persona(tmp_path), min_gap_s=6.0, reaction_gap_s=0.0,
                      clock=lambda: now["t"], mood_classifier=classify)
    items = c.for_addressed("Rocky, hello", now["t"])
    assert [json.loads(i["arguments"])["emotion"] for i in items] == ["thinking"]
    assert calls == ["Rocky, hello"]
    now["t"] += 2
    assert c.for_addressed("Rocky, again", now["t"]) == []          # inside the gap
    assert calls == ["Rocky, hello"]                                 # no mood call inside the gap
    now["t"] += 5
    assert len(c.for_addressed("Rocky, again", now["t"])) == 1       # gap elapsed


def test_a_plain_reply_plays_no_emotion_when_the_call_fails_or_answers_nonsense(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                      mood_classifier=lambda t: (_ for _ in ()).throw(RuntimeError("down")))
    assert c.for_addressed("Rocky, hello", 0.0) == []
    assert json.loads(c.for_addressed("Rocky, what do you think", 0.0)[0]["arguments"]) == {"emotion": "thinking"}
    c2 = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                       mood_classifier=lambda t: "banana")
    assert c2.for_addressed("Rocky, hello", 0.0) == []


def test_a_plain_reply_plays_no_emotion_over_speech_and_spends_no_gap(tmp_path: Path):
    """attentive1 ran 4.28 s over the start of every reply; live, TJ asked what the
    constant emotes were."""
    c = Choreographer(persona(tmp_path), min_gap_s=6.0, reaction_gap_s=0.0, clock=lambda: 0.0)
    assert c.for_addressed("Rocky, hello", 0.0) == [] and c.for_addressed("Rocky, roll me a d20", 1.0) == []
    assert json.loads(c.for_addressed("Rocky, that is hilarious", 2.0)[0]["arguments"]) == {"emotion": "happy"}


def test_addressed_rule_hit_wins_and_skips_the_mood_call(tmp_path: Path):
    """Live, the mood call answered attentive to praise, an opinion question and a
    warning alike, and a valid answer used to beat the rules, so the rules never fired.
    Any rule hit now decides and costs no round trip, a roll included: a roll is the
    commonest thing asked of Rocky, and live the call answered it attentive anyway."""
    asked = []
    def classify(text):
        asked.append(text); return "sad"
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                      mood_classifier=classify)
    def emotion(line):
        return [json.loads(i["arguments"])["emotion"] for i in c.for_addressed(line, 0.0)]
    assert emotion("Rocky, what do you make of the priest?") == ["thinking"]
    assert emotion("Rocky, is that a dragon behind us?") == ["surprised"]
    assert emotion("Rocky, that is hilarious") == ["happy"]
    assert asked == []
    assert emotion("Rocky, poor Tomas is dead") == ["sad"]           # no rule: the call decides
    assert emotion("Rocky, roll me a d20.") == []                    # the roll rule wins too: no emotion
    assert asked == ["Rocky, poor Tomas is dead"]


def test_a_rule_mood_with_no_gesture_is_left_to_the_call(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0)
    assert c.for_addressed("good boy Rocky", 0.0) == []   # rules say proud; this persona has no proud gesture
    c2 = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                       mood_classifier=lambda t: "amused")
    items = c2.for_addressed("good boy Rocky", 0.0)  # a rule mood with no gesture leaves it to the call
    assert [json.loads(i["arguments"]) for i in items] == [{"emotion": "happy"}]


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


def test_table_reaction_rule_hit_skips_the_reaction_call(tmp_path: Path):
    asked = []
    def react(text):
        asked.append(text); return "sad"
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=15.0, clock=lambda: 0.0,
                      reaction_classifier=react)
    assert json.loads(c.for_table("A natural twenty on the goblin chief!", 0.0)[0]["arguments"]) == {"emotion": "happy"}
    assert asked == []


def test_table_reaction_answered_none_moves_no_stamp(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=15.0, clock=lambda: 0.0,
                      reaction_classifier=lambda t: "none")
    assert c.for_table("You walk for an hour along the road.", 0.0) == []
    assert c.last_at == c.last_reaction_at == float("-inf")


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
