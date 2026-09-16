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


# --- the history and revisions speech-to-speech sends, see docs/protocol-notes.md ---

def _msg(role: str, text: str) -> dict:
    part = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": part, "text": text}]}


def _pair(name: str, args: dict, call_id: str, output: dict) -> list[dict]:
    return [{"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args),
             "id": f"fc_{call_id}", "status": "completed"},
            {"type": "function_call_output", "call_id": call_id, "output": json.dumps(output)}]


def _calls(resp) -> list[tuple[str, dict]]:
    out = []
    for l in resp.get_data(as_text=True).splitlines():
        if l.startswith("data: "):
            ev = json.loads(l[6:]); item = ev.get("item") or {}
            if ev["type"] == "response.output_item.done" and item.get("type") == "function_call":
                out.append((item["name"], json.loads(item["arguments"])))
    return out


def _turn(client, items: list[dict]):
    resp = client.post("/v1/responses", json={"model": "familiar", "stream": True, "tools": [], "input": items})
    resp.get_data()  # read the whole stream, as speech-to-speech does
    return resp


SYSTEM = _msg("system", "You are in a spoken conversation.")
IMAGE = {"type": "message", "role": "user",
         "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/AAA="}]}


@pytest.mark.parametrize("stream", [True, False])
def test_gesture_and_idle_calls_in_the_history_never_reach_the_model(env, stream):
    """speech-to-speech records every call the brain sends ahead of the reply text. Left in,
    that history taught Gemma to answer an addressed line with play_emotion and no speech."""
    client, fake, _tmp, _app = env
    kept_before = [SYSTEM, _msg("user", "Understudy, what do you see?"),
                   *_pair("camera", {"question": "scene"}, "call_cam", {"image_attached": True}),
                   IMAGE, _msg("assistant", "Four mortals and a great many dice."),
                   _msg("user", "the door creaks")]
    history = [*kept_before,
               *_pair("idle_do_nothing", {}, "call_idle", {"status": "idle"}),
               _msg("user", "Understudy, how are you?"),
               *_pair("head_tracking", {"enabled": True}, "call_track", {"status": "following"}),
               *_pair("play_emotion", {"emotion": "attentive"}, "call_emo", {"status": "queued"}),
               _msg("assistant", "Content enough."),
               *_pair("move_head", {"direction": "left"}, "call_look", {"status": "moved"}),
               *_pair("stop_emotion", {}, "call_stop", {"status": "stopped"}),
               _msg("user", "Understudy, and the mayor?")]
    resp = client.post("/v1/responses", json={"model": "familiar", "stream": stream, "tools": [], "input": history})
    resp.get_data()
    sent = [r for r in fake.requests if r["path"].endswith("/responses")][-1]["body"]["input"]
    assert sent == [*kept_before, _msg("user", "Understudy, how are you?"), _msg("assistant", "Content enough."),
                    _msg("user", "Understudy, and the mayor?")]


def test_a_revised_line_carries_the_preface_its_superseded_version_took(env):
    """With smart_turn on, speech-to-speech asks for a reply at the first pause and, if the
    player carries on, replaces that user message and asks again. It drops everything the
    first response sent. The revision must still turn tracking on and gesture, even inside
    the live 6 s gesture gap."""
    client, _fake, _tmp, app = env
    first = _calls(_turn(client, [SYSTEM, _msg("user", "Understudy, what's the DC?")]))
    assert first == [("head_tracking", {"enabled": True}), ("play_emotion", {"emotion": "attentive"})]
    revised = [SYSTEM, _msg("user", "Understudy, what's the DC? For the lock, I mean.")]
    assert _calls(_turn(client, revised)) == first
    assert _calls(_turn(client, revised[:1] + [_msg("user", "Understudy, what's the DC? For the lock, I mean it.")])) == first
    # The next line follows a response the client did get, so it is a new turn: tracking is
    # already on and the gesture gap holds.
    delivered = [*revised,
                 *_pair("head_tracking", {"enabled": True}, "call_track", {"status": "following"}),
                 *_pair("play_emotion", {"emotion": "attentive"}, "call_emo", {"status": "queued"}),
                 _msg("assistant", "Fifteen.")]
    assert _calls(_turn(client, [*delivered, _msg("user", "Understudy, and the door?")])) == []
    assert app.extensions["brain_state"].tracking is True


def test_a_revised_line_after_mute_still_switches_tracking_off_and_plays_the_sleep_cue(env):
    client, _fake, _tmp, app = env
    hello = [SYSTEM, _msg("user", "Understudy, hello")]
    assert ("head_tracking", {"enabled": True}) in _calls(_turn(client, hello))
    heard = [*hello, *_pair("head_tracking", {"enabled": True}, "call_track", {"status": "following"}),
             _msg("assistant", "Evening.")]
    client.post("/api/mute")
    expected = [("head_tracking", {"enabled": False}), ("play_emotion", {"emotion": "sleepy"})]
    assert _calls(_turn(client, [*heard, _msg("user", "Natural twenty")])) == expected
    assert _calls(_turn(client, [*heard, _msg("user", "Natural twenty! No wait, nineteen.")])) == expected
    assert app.extensions["brain_state"].tracking is False


def test_a_line_sent_with_no_history_is_never_a_revision(env):
    """speech-to-speech always opens with the system message. A client that sends each line
    bare would otherwise have every line taken for a revision of the one before."""
    client, _fake, _tmp, _app = env
    assert _calls(_turn(client, [_msg("user", "Understudy, what's the DC?")]))
    assert _calls(_turn(client, [_msg("user", "Understudy, what's the DC? For the lock.")])) == []


def test_the_same_history_long_after_is_not_a_revision(env, monkeypatch):
    """A reconnect starts from the same system message, so the history alone cannot tell a
    revision from a first line in a new connection; only a request soon after can be one."""
    from familiar_brain import app as app_module
    monkeypatch.setattr(app_module, "REVISION_WINDOW_S", 0.0)
    client, _fake, _tmp, _app = env
    assert _calls(_turn(client, [SYSTEM, _msg("user", "Understudy, what's the DC?")]))
    assert _calls(_turn(client, [SYSTEM, _msg("user", "Understudy, what's the DC? For the lock.")])) == []


ROBOT_TOOLS = [{"type": "function", "name": n, "parameters": {"type": "object", "properties": {}}}
               for n in ("move_head", "play_emotion", "stop_emotion", "idle_do_nothing", "head_tracking", "camera")]


@pytest.mark.parametrize("stream", [True, False])
def test_gesture_tools_are_never_offered_to_the_model(env, stream):
    """Offered them, Gemma answered an addressed line with a bare move_head, or spoke the
    call aloud as text. The brain gestures on its own, so the model only needs the camera."""
    client, fake, _tmp, _app = env
    client.post("/v1/responses", json={"model": "familiar", "stream": stream, "tools": ROBOT_TOOLS,
                                       "input": [SYSTEM, _msg("user", "Understudy, look at me")]}).get_data()
    offered = {t["name"] for t in [r for r in fake.requests if r["path"].endswith("/responses")][-1]["body"]["tools"]}
    assert "camera" in offered and {"roll", "note"} <= offered
    assert not offered & {"move_head", "play_emotion", "stop_emotion", "idle_do_nothing", "head_tracking"}


def test_an_addressed_turn_that_came_back_without_words_is_asked_again(env):
    client, fake, _tmp, app = env
    bare = {"id": "fc_x", "type": "function_call", "call_id": "call_x", "name": "move_head",
            "arguments": '{"direction":"front"}', "status": "completed"}
    fake.responses_events.extend([
        [{"type": "response.output_item.done", "output_index": 0, "item": bare},
         {"type": "response.completed", "response": {"id": "r1", "output": [bare]}}],
        [{"type": "response.output_text.delta", "delta": "I am looking."},
         {"type": "response.completed", "response": {"id": "r2", "output": []}}],
    ])
    resp = _turn(client, [SYSTEM, _msg("user", "Understudy, look at me")])
    assert "response.output_text.delta" in _types(resp)
    assert len([r for r in fake.requests if r["path"].endswith("/responses")]) == 2
    assert app.extensions["brain_state"].last_spoke_at is not None


@pytest.mark.parametrize("route", ["/api/mute", "/api/mute_and_stop_logging"])
def test_a_mute_closes_the_follow_up_window(env, route):
    """Live: named and answered, muted, unmuted, and the next table line inside the old
    window was answered uninvited."""
    client, _fake, _tmp, _app = env
    assert "response.output_text.delta" in _types(_turn(client, [_msg("user", "Understudy, roll me a d20")]))
    client.post(route)
    assert "response.output_text.delta" not in _types(_turn(client, [_msg("user", "Understudy, hello?")]))
    client.post("/api/unmute")
    # The classifier answers IGNORE (the fake's default) once the window is closed.
    assert "response.output_text.delta" not in _types(_turn(client, [_msg("user", "moving on then")]))


def test_a_mute_disarms_a_pending_speak_now(env):
    client, _fake, _tmp, _app = env
    client.post("/api/speak_now")
    client.post("/api/mute")
    client.post("/api/unmute")
    assert "response.output_text.delta" not in _types(_turn(client, [_msg("user", "moving on then")]))


def test_the_tool_follow_up_prefetch_is_answered_empty_and_changes_nothing(env):
    """Live, the prefetch after every gestured reply was gated as a new ADDRESSED turn: a
    second mood call and gesture, retrieval and a full Gemma generation nobody heard, run
    while the reply was still being synthesised."""
    client, fake, tmp, app = env
    client.post("/api/session/start")
    line = _msg("user", "Understudy, hello")
    _turn(client, [SYSTEM, line])
    bs = app.extensions["brain_state"]
    snapshot = lambda: (list(bs.recent_segments), bs.last_named_at, bs.follow_ups, bs.tracking,
                        bs.last_spoke_at, choreo_calls())
    choreo_calls = lambda: len([r for r in fake.requests if r["path"].endswith("/chat/completions")])
    before, requests_before = snapshot(), len(fake.requests)
    log_before = next((tmp / "sessions").glob("*.jsonl")).read_text()
    prefetch = [SYSTEM, line, *_pair("head_tracking", {"enabled": True}, "call_t", {"status": "following"}),
                *_pair("play_emotion", {"emotion": "attentive"}, "call_e", {"status": "queued"}),
                _msg("assistant", "Evening.")]
    resp = _turn(client, prefetch)
    assert _types(resp)[-1] == "response.completed" and _calls(resp) == []
    assert "response.output_text.delta" not in _types(resp)
    assert len(fake.requests) == requests_before and snapshot() == before
    assert next((tmp / "sessions").glob("*.jsonl")).read_text() == log_before
    # A real line after the reply is still a turn.
    assert "response.output_text.delta" in _types(_turn(client, [*prefetch, _msg("user", "Understudy, and the door?")]))


COMPACTION = [{"type": "message", "role": "system", "content": [{"type": "input_text", "text": "You are a conversation memory compressor."}]},
              _msg("user", "Summarize the following conversation.  Return only the JSON object.\n\n"
                           "--- CONVERSATION START ---\nUser: Understudy, help me think through the heist.\n\n"
                           "Assistant: Start with the guards.\n--- CONVERSATION END ---")]


def test_history_compaction_goes_upstream_untouched_and_touches_no_state(env):
    """Gated, a transcript that named the familiar opened the follow-up window and "help me
    think" switched on thinking mode; answered in persona, the summary was not JSON."""
    client, fake, tmp, app = env
    client.post("/api/session/start")
    fake.responses_json.append({"id": "resp_c", "object": "response", "status": "completed", "created_at": 0,
                                "model": "familiar", "output": [{"type": "message", "role": "assistant",
                                "content": [{"type": "output_text", "text": "{\"user_summary\": \"x\"}"}]}]})
    resp = client.post("/v1/responses", json={"model": "familiar", "input": COMPACTION})
    assert resp.get_json()["id"] == "resp_c"
    sent = fake.requests[-1]["body"]
    assert sent["input"] == COMPACTION and "instructions" not in sent and "tools" not in sent
    assert [r for r in fake.requests if r["path"].endswith("/chat/completions")] == []
    bs = app.extensions["brain_state"]
    assert bs.last_named_at is None and bs.mode == "familiar" and bs.recent_segments == []
    assert next((tmp / "sessions").glob("*.jsonl")).read_text() == ""


def test_the_mood_call_before_a_reply_waits_at_most_half_a_second(env, monkeypatch):
    """It runs before the first word. Under GPU contention it took 600 to 900 ms, and with
    the gate's 1.5 s budget a reply could wait that long for a facial expression."""
    import requests as rq
    client, _fake, _tmp, app = env
    real, seen = rq.post, []

    def spy(url, *a, **kw):
        if url.endswith("/chat/completions"):
            seen.append((kw["json"]["messages"][0]["content"], kw.get("timeout")))
        return real(url, *a, **kw)

    monkeypatch.setattr(rq, "post", spy)
    _turn(client, [_msg("user", "Understudy, hello there")])
    app.extensions["brain_state"].last_named_at -= 60   # past the window, so the gate asks
    _turn(client, [_msg("user", "the innkeeper wipes down the long oak bar")])
    mood = [t for prompt, t in seen if prompt.startswith("You choose the facial expression")]
    gate = [t for prompt, t in seen if prompt.startswith("You are a strict filter")]
    assert mood and all(t <= 0.5 for t in mood)
    assert gate == [1.0]                        # the gate keeps its configured budget
