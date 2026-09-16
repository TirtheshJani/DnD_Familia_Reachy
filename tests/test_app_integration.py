import json
from pathlib import Path
import pytest
from familiar_brain import indexing
from familiar_brain.app import create_app
from familiar_brain.retrieval import Store
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm_server import FakeLLM

FIX = Path(__file__).parent / "fixtures" / "mock_session.jsonl"


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    fake = FakeLLM()
    # Without this the brain builds a real BgeEmbedder and the first addressed turn
    # spends about twelve seconds loading it, which runs the clock past the follow-up
    # window and makes this test fail on the follow-up row. No test should load a
    # 130 MB model or depend on the network.
    monkeypatch.setattr(indexing, "open_store",
                        lambda cfg: Store(tmp_path / "k.db", embedder=FakeEmbedder()))
    persona = tmp_path / "persona.yaml"
    persona.write_text("name: Understudy\naliases: [understudy, little one]\ntts_speaker: Aiden\nmanner: dry\n"
                       "gestures:\n"
                       "  attentive: play_emotion:attentive\n"
                       "  amused: play_emotion:happy\n"
                       "  alarmed: play_emotion:surprised\n"
                       "  thoughtful: play_emotion:thinking\n"
                       "  sad: play_emotion:sad\n"
                       "  scared: play_emotion:scared\n"
                       "  asleep: play_emotion:sleepy\n"
                       "  awake: play_emotion:welcoming\n")
    cfg = tmp_path / "brain.toml"
    # as_posix: a raw Windows path would put \U and \s escapes inside a TOML basic
    # string, which tomllib rejects. Forward slashes parse on both platforms.
    cfg.write_text(f"""
[server]
host = "127.0.0.1"
port = 0
[upstream]
base_url = "{fake.base_url}"
generate_slot = 0
classify_slot = 1
classify_timeout_s = 1.0
[gate]
follow_up_window_s = 8
[paths]
persona = "{persona.as_posix()}"
sessions_dir = "{(tmp_path / 'sessions').as_posix()}"
[knowledge]
db_path = "{(tmp_path / 'k.db').as_posix()}"
notes_dir = "{(tmp_path / 'notes').as_posix()}"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
""")
    app = create_app(cfg)
    app.config["TESTING"] = True
    yield app.test_client(), fake, tmp_path, app
    fake.close()


def _types(resp) -> list[str]:
    return [json.loads(l[6:])["type"] for l in resp.get_data(as_text=True).splitlines() if l.startswith("data: ")]


def _names(resp) -> list[str]:
    out = []
    for l in resp.get_data(as_text=True).splitlines():
        if l.startswith("data: "):
            ev = json.loads(l[6:]); item = ev.get("item") or {}
            if ev["type"] == "response.output_item.done" and item.get("type") == "function_call":
                out.append(item["name"])
    return out


def test_mock_session_speaks_only_when_addressed(env):
    client, fake, tmp, app = env
    client.post("/api/session/start")
    for line in FIX.read_text().splitlines():
        row = json.loads(line)
        if row.get("after_window"):  # simulate silence long enough to close the follow-up window
            bs = app.extensions["brain_state"]
            bs.last_spoke_at -= 60
            # The window is anchored to naming now, so backdating speech alone leaves
            # it wide open. Both stamps have to move.
            bs.last_named_at -= 60
        if "classifier" in row:
            fake.chat_replies.append(row["classifier"])
        fake.responses_events.append([
            {"type": "response.output_text.delta", "delta": "In character reply."},
            {"type": "response.completed", "response": {"id": "r", "output": []}},
        ])
        resp = client.post("/v1/responses", json={"model": "familiar", "stream": True,
                                                    "input": [{"role": "user", "content": row["text"]}], "tools": []})
        assert resp.status_code == 200
        names = _names(resp)
        if row["expect"] == "ADDRESSED":
            assert "idle_do_nothing" not in names and "response.output_text.delta" in _types(resp)
        else:
            assert "response.output_text.delta" not in _types(resp)
            expected_motion = {"AMBIENT": {"play_emotion"},
                               "IGNORE": {"idle_do_nothing", "head_tracking", "move_head", "play_emotion"}}
            assert set(names) <= expected_motion[row["expect"]] and names
    # every utterance was logged regardless of decision
    sessions = list((tmp / "sessions").glob("*.jsonl"))
    assert len(sessions) == 1
    logged = [json.loads(l) for l in sessions[0].read_text().splitlines()]
    assert len(logged) == 8 and [r["decision"] for r in logged] == [json.loads(l)["expect"] for l in FIX.read_text().splitlines()]


def test_mute_blocks_and_state_reports(env):
    client, fake, _, _app = env
    client.post("/api/session/start")
    client.post("/api/mute")
    resp = client.post("/v1/responses", json={"input": [{"role": "user", "content": "Understudy!"}], "stream": True})
    assert _names(resp) == ["play_emotion"]
    assert client.get("/api/state").get_json()["muted"] is True
    client.post("/api/unmute")
    assert client.get("/api/state").get_json()["muted"] is False


def test_persona_instructions_and_tools_reach_upstream(env):
    client, fake, _, _app = env
    client.post("/api/session/start")
    # Read the stream as speech-to-speech does. The test client reads only the first chunk
    # on its own, and since Plan 3 that chunk is the gesture preface, sent before the
    # upstream call is made.
    client.post("/v1/responses", json={"input": [{"role": "user", "content": "Understudy, hello"}],
                                       "stream": True, "tools": []}).get_data()
    body = fake.requests[-1]["body"]
    assert "Understudy" in body["instructions"] and "Dungeon Master" in body["instructions"]
    assert {"roll", "initiative", "note"} <= {t["name"] for t in body["tools"]}


# --- deviation tests, see docs/deviations.md ---

def test_warmup_shaped_request_gets_a_json_response_not_sse(env):
    """ResponsesApiModelHandler.warmup() posts without `stream` and the openai SDK
    parses the body as a Response object. Answering with SSE breaks pipeline startup."""
    client, _fake, _tmp, _app = env
    resp = client.post("/v1/responses", json={
        "model": "familiar",
        "input": [
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "You are a helpful assistant"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
        ],
    })
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("application/json")
    body = resp.get_json()
    assert body["object"] == "response" and body["status"] == "completed"
    assert body["model"] == "familiar" and isinstance(body["output"], list)


def test_non_streaming_addressed_turn_proxies_upstream(env):
    """The compaction helper also calls without `stream` and reads output_text, so an
    addressed non-streaming turn must reach llama-server rather than return idle."""
    client, fake, _tmp, _app = env
    client.post("/api/session/start")
    fake.responses_json.append({
        "id": "resp_up", "object": "response", "status": "completed", "created_at": 0,
        "model": "familiar",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "A damp, iron smell."}]}],
    })
    resp = client.post("/v1/responses", json={
        "model": "familiar",
        "input": [{"role": "user", "content": "Understudy, what do you smell?"}],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["id"] == "resp_up"
    assert body["output"][0]["content"][0]["text"] == "A damp, iron smell."
    assert fake.requests[-1]["body"]["stream"] is False


def test_warmup_does_not_open_the_follow_up_window(env):
    """The client's warmup probe says "Hello" at startup. Observed live: the classifier
    called that ADDRESSED, the brain answered, and marking it as speech opened the
    follow-up window, so the very first real utterance at the table was treated as a
    follow-up and answered too. A warmup probe is not speech at the table."""
    client, fake, _tmp, app = env
    client.post("/api/session/start")
    fake.chat_replies.append("ADDRESSED")  # what the live classifier actually said
    fake.responses_json.append({"id": "resp_warm", "object": "response", "status": "completed",
                                "created_at": 0, "model": "familiar", "output": []})
    client.post("/v1/responses", json={
        "model": "familiar",
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": "Hello"}]}]})
    assert app.extensions["brain_state"].last_spoke_at is None, \
        "a non-spoken warmup turn must not arm the follow-up window"
    # so an unrelated line straight afterwards still goes through the real gate
    fake.chat_replies.append("IGNORE")
    resp = client.post("/v1/responses", json={
        "input": [{"role": "user", "content": "You descend into the cistern."}], "stream": True})
    assert _names(resp) == ["idle_do_nothing"]


def test_session_end_without_a_session_is_a_clean_error(env):
    """SessionLog.path raises RuntimeError with no active session, which the plan's
    handler turns into a 500."""
    client, _fake, _tmp, _app = env
    resp = client.post("/api/session/end")
    assert resp.status_code == 409
    assert "no active session" in resp.get_json()["error"]
