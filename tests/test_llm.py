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


def test_classify_unparseable_raises(fake):
    fake.chat_replies.append("well, maybe")
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    with pytest.raises(ValueError):
        c.classify(["x"], persona_name="U")


def test_classify_timeout_raises(fake):
    c = LLMClient(base_url="http://127.0.0.1:9", classify_slot=1, classify_timeout_s=0.2)
    with pytest.raises(Exception):
        c.classify(["x"], persona_name="U")
