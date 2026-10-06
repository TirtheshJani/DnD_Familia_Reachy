import pytest
from familiar_brain.gate import Decision
from familiar_brain.llm import LLMClient
from tests.fakes.llm_server import FakeLLM


@pytest.fixture
def fake():
    f = FakeLLM(); yield f; f.close()


def test_classify_parses_token_and_pins_slot(fake):
    fake.chat_replies.append("AMBIENT")
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    assert c.classify(["Understudy is cute", "what do you think"], persona_name="Understudy") == Decision.AMBIENT
    body = fake.requests[-1]["body"]
    assert body["id_slot"] == 1 and body["max_tokens"] <= 4 and body["stream"] is False


def test_classify_judges_only_the_last_line(fake):
    """Design 5 layer 3 says to send the last four segments. Measured on twelve labelled
    lines, that context makes this model far worse, not better: once any earlier segment
    names the familiar it reads everything after as continuing the conversation. Four
    segments scored 4/12 and would have spoken uninvited 7 times; the last line alone
    scored 11/12 and never spoke uninvited. Layers 1 and 2 already cover 'it was named'
    and 'it just spoke', so the classifier does not need the history."""
    fake.chat_replies.append("IGNORE")
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    c.classify(["Understudy, roll me a d20.", "Hello", "the door creaks open"],
               persona_name="Understudy")
    sent = fake.requests[-1]["body"]["messages"][-1]["content"]
    assert "the door creaks open" in sent
    assert "Understudy, roll me a d20." not in sent, "earlier context biases the verdict"
    assert "Hello" not in sent


def test_classify_disables_reasoning(fake):
    """The classifier asks for one token on a 600 ms budget. Measured against the live
    model, the same call took 286 ms with reasoning left on and 59 ms with it off, so
    the margin against a contended slot is worth having."""
    fake.chat_replies.append("AMBIENT")
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    c.classify(["Natural twenty!"], persona_name="Understudy")
    assert fake.requests[-1]["body"]["reasoning_effort"] == "none"


def test_classify_only_tells_ambient_from_ignore(fake):
    """Rocky replies only when spoken to, which the gate decides by its name, the window
    after a naming, or Speak Now. The classifier is left only the reaction question, so
    the instruction no longer offers ADDRESSED and an ADDRESSED answer does not parse."""
    from familiar_brain.llm import CLASSIFY_INSTRUCTION
    assert "ADDRESSED" not in CLASSIFY_INSTRUCTION
    assert "AMBIENT" in CLASSIFY_INSTRUCTION and "IGNORE" in CLASSIFY_INSTRUCTION
    fake.chat_replies.append("IGNORE")
    fake.chat_replies.append("ADDRESSED")
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    assert c.classify(["what's behind the door?"], persona_name="Rocky") == Decision.IGNORE
    with pytest.raises(ValueError):
        c.classify(["Rocky, what's behind the door?"], persona_name="Rocky")


def test_classify_unparseable_raises(fake):
    fake.chat_replies.append("well, maybe")
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    with pytest.raises(ValueError):
        c.classify(["x"], persona_name="U")


def test_classify_timeout_raises(fake):
    c = LLMClient(base_url="http://127.0.0.1:9", classify_slot=1, classify_timeout_s=0.2)
    with pytest.raises(Exception):
        c.classify(["x"], persona_name="U")


def test_mood_is_one_token_on_the_classify_slot_and_validated(fake):
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    fake.chat_replies.append("Thoughtful.")
    assert c.mood("Rocky, what do you make of him?", moods=("attentive", "thoughtful")) == "thoughtful"
    body = fake.requests[-1]["body"]
    assert body["id_slot"] == 1 and body["max_tokens"] == 4 and body["reasoning_effort"] == "none"
    assert "attentive, thoughtful" in body["messages"][0]["content"]
    assert "spoken to" in body["messages"][0]["content"]
    fake.chat_replies.append("banana")
    assert c.mood("anything", moods=("attentive", "thoughtful")) is None


def test_mood_prompt_names_what_each_mood_answers_and_keeps_attentive_for_plain_requests(fake):
    """Scored live on twelve labelled addressed lines. The wording that ended "Choose
    attentive when nothing else clearly fits" got 6 of 12: attentive to praise, an
    opinion question, a warning and a death. Saying what each mood answers got 11 of 12,
    and still answered attentive to the roll, the greeting and the question of fact."""
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    fake.chat_replies.append("proud")
    c.mood("Good job Rocky, that was clever.", moods=("attentive", "proud", "sad"))
    content = fake.requests[-1]["body"]["messages"][0]["content"]
    assert "attentive: a plain request, a greeting, a question of fact" in content
    assert "proud: praise or thanks" in content and "sad: bad news" in content
    assert "nothing else clearly fits" not in content


def test_reaction_prompt_is_selectable_and_none_is_a_valid_answer(fake):
    from familiar_brain.llm import REACTION_INSTRUCTION
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    fake.chat_replies.append("none")
    moods = ("amused", "alarmed", "sad", "scared", "none")
    assert c.mood("You walk for an hour along the road.", moods=moods, prompt=REACTION_INSTRUCTION) == "none"
    content = fake.requests[-1]["body"]["messages"][0]["content"]
    assert "not spoken to it" in content and "Most lines are none" in content
    fake.chat_replies.append("Alarmed")
    assert c.mood("The priest pulls back his hood: it is the king.", moods=moods, prompt=REACTION_INSTRUCTION) == "alarmed"


def test_mood_reads_the_first_word_and_never_turns_a_negation_into_a_gesture(fake):
    from familiar_brain.llm import REACTION_INSTRUCTION
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    moods = ("amused", "alarmed", "sad", "scared", "none")
    for reply, want in [("**Sad**", "sad"), ("alarmed: a shocking", "alarmed"), ("Sad\n\nThe line", "sad"),
                        ("not sad", None), ("no reaction", None), ("Sadly, none", None), ("", None)]:
        fake.chat_replies.append(reply)
        assert c.mood("x", moods=moods, prompt=REACTION_INSTRUCTION) == want, reply


def test_reaction_prompt_tells_a_victory_from_a_joke():
    """cheer (success) is a victory for the party; amused (happy) keeps only the funny."""
    from familiar_brain.llm import REACTION_INSTRUCTION
    assert "cheer: a victory or a triumph for the party" in REACTION_INSTRUCTION
    assert "amused: the line is funny.\n" in REACTION_INSTRUCTION


def test_complete_refuses_a_reply_cut_off_at_max_tokens(fake):
    """A summary or campaign merge stopped at max_tokens loses its trailing headings (Open
    threads, Memorable lines, Loot), yet read as a whole reply it was written and indexed as
    complete, and catch-up never retried it because the .summary.md existed."""
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    fake.chat_replies.append("## Scenes\n- The crypt\n## Open thr")
    fake.chat_finish.append("length")
    with pytest.raises(ValueError, match="max_tokens"):
        c.complete([{"role": "user", "content": "summarise"}], max_tokens=1200)
    fake.chat_replies.append("## Scenes\n- The crypt\n")
    assert c.complete([{"role": "user", "content": "summarise"}]) == "## Scenes\n- The crypt\n"
