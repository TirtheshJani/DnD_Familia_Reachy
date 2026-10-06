import json
from pathlib import Path

from familiar_brain.motion import (
    REACTION_MOODS, Choreographer, table_mood, addressed_mood, tracking_item,
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
  cheer: play_emotion:success
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


def test_reaction_moods_are_the_narration_reactions_and_cheer():
    assert REACTION_MOODS == ("amused", "cheer", "alarmed", "sad", "scared")


def test_addressed_mood_rules_say_none_when_nothing_matches():
    """Like table_mood: None means no rule matched, so the caller can tell a roll, which
    the rules answer attentive, from a line the rules know nothing about."""
    assert addressed_mood("Rocky, roll me a d20") == "attentive"
    assert addressed_mood("Rocky, what do you think of him?") == "thoughtful"
    assert addressed_mood("good boy Rocky") == "proud"
    assert addressed_mood("Rocky, and the docks?") is None


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


def test_tracking_item():
    t = tracking_item(True)
    assert t["name"] == "head_tracking" and json.loads(t["arguments"]) == {"enabled": True}


def test_choreographer_asks_the_classifier_when_no_rule_matches_and_rate_limits(tmp_path: Path):
    now = {"t": 100.0}
    calls = []
    def classify(text):
        calls.append(text); return "thoughtful"
    c = Choreographer(persona(tmp_path), min_gap_s=6.0, reaction_gap_s=0.0,
                      clock=lambda: now["t"], mood_classifier=classify)
    items = c.for_addressed("Rocky, and the docks?", now["t"])
    assert [json.loads(i["arguments"])["emotion"] for i in items] == ["thinking"]
    assert calls == ["Rocky, and the docks?"]
    now["t"] += 2
    assert c.for_addressed("Rocky, again", now["t"]) == []          # inside the gap
    assert calls == ["Rocky, and the docks?"]                        # no mood call inside the gap
    now["t"] += 5
    assert len(c.for_addressed("Rocky, again", now["t"])) == 1       # gap elapsed


def test_a_plain_reply_plays_no_emotion_when_the_call_fails_or_answers_nonsense(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                      mood_classifier=lambda t: (_ for _ in ()).throw(RuntimeError("down")))
    assert c.for_addressed("Rocky, and the docks?", 0.0) == []
    assert json.loads(c.for_addressed("Rocky, what do you think", 0.0)[0]["arguments"]) == {"emotion": "thinking"}
    c2 = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                       mood_classifier=lambda t: "banana")
    assert c2.for_addressed("Rocky, and the docks?", 0.0) == []


def test_a_plain_reply_plays_no_emotion_over_speech_and_spends_no_gap(tmp_path: Path):
    """attentive1 ran 4.28 s over the start of every reply; live, TJ asked what the
    constant emotes were."""
    c = Choreographer(persona(tmp_path), min_gap_s=6.0, reaction_gap_s=0.0, clock=lambda: 0.0)
    assert c.for_addressed("Rocky, and the docks?", 0.0) == [] and c.for_addressed("Rocky, roll me a d20", 1.0) == []
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


# --- 2026-10-02: more expression from cheap rules (deviation P15) ---

def test_table_rules_catch_shock_at_the_table():
    for line in ("Oh no!", "oh no", "Oh no. He's going for the wizard", "No way!", "What? No way.",
                 "Oh my god", "oh my gods, the lich", "Oh my gosh.", "what the hell was that",
                 "What the heck?", "Are you kidding me?", "you're kidding", "Holy crap!"):
        assert table_mood(line) == "alarmed", line


def test_table_shock_rules_leave_ordinary_table_talk_alone():
    """"No way" is a shock only as an exclamation of its own; inside a sentence it is the
    commonest thing said about a chasm. "What the" needs its oath after it."""
    for line in ("there's no way to get across", "no way out of the vault", "Is there no way in?",
                 "Oh, nothing, I was just looking.", "what the door says is in Elvish",
                 "what the innkeeper wants is gold", "Oh, not again.", "are you serious about the heist plan",
                 "Oh my, the goddess is beautiful"):
        assert table_mood(line) is None, line


def test_table_rules_cheer_a_victory():
    for line in ("Let's go!", "LETS GOOO", "Yes!", "yes yes yes!", "Hell yes!", "We did it!", "we did it",
                 "We won!", "we won.", "Nailed it!", "Huzzah!", "Victory!", "Woohoo!"):
        assert table_mood(line) == "cheer", line


def test_table_victory_rules_leave_plans_and_answers_alone():
    """"Let's go to the tavern" is a plan and "Let's go." is moving on; only an elongated
    or exclaimed one cheers. A plain "Yes." or "Yes, I attack" answers a question. "Killed
    it" is left out: at this table it is as likely the wolf that killed the horse."""
    for line in ("let's go to the tavern", "Okay, let's go.", "let's go", "Yes.", "Yes, I attack the goblin",
                 "yes I want to", "we did it yesterday", "we did it wrong", "we won't make it",
                 "we won initiative last time", "I killed it", "the wolf killed it"):
        assert table_mood(line) is None, line


def test_table_rules_lean_in_when_combat_starts_and_mourn_a_little_more():
    assert table_mood("Roll initiative!") == "attentive"
    assert table_mood("everyone roll for initiative") == "attentive"
    assert table_mood("what's my initiative bonus again") is None
    assert table_mood("Rest in peace, Tomas.") == "sad" and table_mood("that's so sad") == "sad"
    assert table_mood("she didn't make it") == "sad"
    assert table_mood("[laughter]") == "amused" and table_mood("(laughs)") == "amused"
    assert table_mood("A shape moves behind the altar and laughs.") is None   # narration: left to the call


def test_table_rules_keep_their_order_when_a_line_says_two_things():
    """The first rule wins, so the specific beats the general: a nat 1 or a downed friend
    said with an "oh no" keeps its own reaction."""
    assert table_mood("oh no, nat 1") == "alarmed"
    assert table_mood("Oh no, he went down!") == "scared"
    assert table_mood("oh my god hahaha") == "amused"
    assert table_mood("Natural twenty! Let's go!") == "amused"


def test_addressed_rules_greet_and_see_off_but_only_a_line_that_is_all_greeting():
    for line in ("Hi Rocky!", "Rocky, hello", "Hello, Rocky.", "Good evening, Rocky!", "Welcome back, Rocky",
                 "Hello Rocky, how are you?", "Good morning Rocky, how's it going?"):
        assert addressed_mood(line) == "awake", line
    for line in ("Goodnight, Rocky", "Bye Rocky!", "Rocky, see you next session.", "Good night Rocky", "goodbye rocky"):
        assert addressed_mood(line) == "awake", line
    # "Hey" calls Rocky's attention before nearly every request; as a greeting it would
    # bring back the emote on every reply that TJ asked about on 2026-09-16.
    for line in ("Hey Rocky, roll me a d20", "Hey Rocky", "Hello Rocky, roll me a d20",
                 "Rocky, say hello to the new player", "Rocky, what does welcome mean in Elvish?",
                 "Rocky, is the inn called the Goodnight Rest?", "Rocky, see you at the docks, I mean, tell me about the docks"):
        assert addressed_mood(line) != "awake", line


def test_a_greeting_plays_welcoming_and_a_plain_reply_still_plays_nothing(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0)
    assert [json.loads(i["arguments"]) for i in c.for_addressed("Hi Rocky!", 0.0)] == [{"emotion": "welcoming"}]
    assert [json.loads(i["arguments"]) for i in c.for_addressed("Goodnight, Rocky", 1.0)] == [{"emotion": "welcoming"}]
    for line in ("Rocky, and the docks?", "Hey Rocky, roll me a d20", "Rocky, who is the mayor?"):
        assert c.for_addressed(line, 2.0) == [], line


def test_a_victory_at_the_table_plays_success(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0)
    assert [json.loads(i["arguments"]) for i in c.for_table("We did it!", 0.0)] == [{"emotion": "success"}]


def test_an_ambient_burst_asks_the_reaction_call_however_short(tmp_path: Path):
    """The gate answers AMBIENT only for a burst of excitement or dismay, which is short
    by nature, so the five word floor that keeps narration chatter off the call does not
    apply."""
    asked = []
    def react(text):
        asked.append(text); return "scared"
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                      reaction_classifier=react)
    assert [json.loads(i["arguments"]) for i in c.for_table("ugh seriously", 0.0, ambient=True)] == [{"emotion": "scared"}]
    assert asked == ["ugh seriously"]
    assert c.for_table("ugh seriously", 1.0) == [] and asked == ["ugh seriously"]   # IGNORE: still the floor


def test_an_ambient_burst_falls_back_to_surprised_when_the_call_fails(tmp_path: Path):
    """Live the reaction call timed out at 1.5 s on some lines. Surprised reads as engaged
    for a cheer and a groan alike, where amused would grin at a friend going down and sad
    would mope at a natural twenty. An explicit none is an answer, not a failure: see
    test_an_ambient_line_the_reaction_call_answers_none_plays_nothing."""
    for react in (lambda t: "banana",
                  lambda t: (_ for _ in ()).throw(RuntimeError("ReadTimeout")), None):
        c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                          reaction_classifier=react)
        assert [json.loads(i["arguments"]) for i in c.for_table("ugh seriously", 0.0, ambient=True)] == [
            {"emotion": "surprised"}]


def test_an_ambient_burst_respects_both_gaps(tmp_path: Path):
    asked = []
    def react(text):
        asked.append(text); return None          # no usable answer, so the fallback plays
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=10.0, clock=lambda: 0.0,
                      reaction_classifier=react)
    assert len(c.for_table("ugh seriously", 0.0, ambient=True)) == 1
    assert c.for_table("come on", 5.0, ambient=True) == [] and asked == ["ugh seriously"]   # no call in the gap
    assert len(c.for_table("come on", 11.0, ambient=True)) == 1


# --- 2026-10-02: review fixes to the P15 rules (deviation P16) ---

def test_an_ambient_line_the_reaction_call_answers_none_plays_nothing(tmp_path: Path):
    """The classifier labels some Dungeon Master narration AMBIENT; in data/sessions one
    of the two AMBIENT lines was this one. The reaction call saying none is right, and the
    fallback used to override it with surprised."""
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                      reaction_classifier=lambda t: "none")
    assert c.for_table("You descend into the cistern. Water drips somewhere in the dark.", 100.0,
                       ambient=True) == []
    assert c.last_at == c.last_reaction_at == float("-inf")


def test_an_ambient_reaction_with_no_gesture_falls_back_to_surprised(tmp_path: Path):
    """persona.yaml is gitignored, so a copy made before cheer existed has no gesture for it."""
    p = tmp_path / "old.yaml"
    p.write_text(PERSONA.replace("  cheer: play_emotion:success\n", ""))
    c = Choreographer(load_persona(p), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                      reaction_classifier=lambda t: "cheer")
    assert [json.loads(i["arguments"]) for i in c.for_table("We got him!", 100.0, ambient=True)] == [
        {"emotion": "surprised"}]


def test_table_rules_read_only_the_end_of_a_long_segment():
    """Live, 70 words of merged small talk: the "Oh no," is 33 words from the end, chatter
    before the line that matters, as the gate's own run-on rule says."""
    line = ("I might come over there because I'm clearly crowding these things. No, I can move "
            "this out. It makes it easier. You're already sourced. I'm already moving. Okay. "
            "Whatever makes life easier. I'm I'm still uh I don't shuttle get I also will not "
            "shuttle. Oh no, how are you? How are you? Hey, how's it going? Good to you. Oh "
            "thanks. Sam? Yeah. Yeah. Yeah, you want this? Okay.")
    assert table_mood(line) is None
    assert table_mood(" ".join(["we walk on for a while"] * 6) + " natural twenty!") == "amused"


def test_shock_needs_an_exclamation_not_a_question_or_a_greeting():
    for line in ("Oh no, how are you?", "What the heck is a sennight?",
                 "Are you kidding me, it's my turn already?", "Holy crap, it's ten o'clock."):
        assert table_mood(line) is None, line
    for line in ("What the heck?", "what the hell was that", "Are you kidding me?", "Holy crap!",
                 "No! No no no."):
        assert table_mood(line) == "alarmed", line


def test_victory_is_an_exclamation_not_a_name_or_narration():
    for line in ("Victory Road leads north out of the city.", "You see the Victory Arch in the distance.",
                 "Who's holding the victory banner?", "The orcs roar in victory as your friend falls",
                 "what were the victory conditions again", "the crowd shouts hooray for the king"):
        assert table_mood(line) is None, line
    for line in ("Victory!", "victory", "Hooray!", "Huzzah, we live"):
        assert table_mood(line) == "cheer", line


def test_unpunctuated_triumphs_still_cheer():
    for line in ("YES YES YES", "Fuck yes", "We killed the dragon!", "we got him!"):
        assert table_mood(line) == "cheer", line
    assert table_mood("I rolled a twenty!") == "amused" and table_mood("rolled a 20") == "amused"


def test_initiative_leans_in_without_costing_the_first_crit_its_reaction(tmp_path: Path):
    assert table_mood("Do I need to roll initiative or is this a surprise round?") is None
    assert table_mood("Should we roll for initiative?") is None
    c = Choreographer(persona(tmp_path), min_gap_s=6.0, reaction_gap_s=10.0, clock=lambda: 0.0)
    assert [json.loads(i["arguments"]) for i in c.for_table("Roll for initiative.", 0.0)] == [
        {"emotion": "attentive"}]
    assert [json.loads(i["arguments"]) for i in c.for_table("Nat twenty!", 8.0)] == [{"emotion": "happy"}]
