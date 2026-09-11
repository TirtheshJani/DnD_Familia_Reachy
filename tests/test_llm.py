import pytest
from familiar_brain.gate import Decision
from familiar_brain.llm import LLMClient
from tests.fakes.llm_server import FakeLLM


@pytest.fixture
def fake():
    f = FakeLLM(); yield f; f.close()


def test_classify_parses_token_and_pins_slot(fake):
    fake.chat_replies.append("ADDRESSED")
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    assert c.classify(["Understudy is cute", "what do you think"], persona_name="Understudy") == Decision.ADDRESSED
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


def test_classify_unparseable_raises(fake):
    fake.chat_replies.append("well, maybe")
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    with pytest.raises(ValueError):
        c.classify(["x"], persona_name="U")


def test_classify_timeout_raises(fake):
    c = LLMClient(base_url="http://127.0.0.1:9", classify_slot=1, classify_timeout_s=0.2)
    with pytest.raises(Exception):
        c.classify(["x"], persona_name="U")
