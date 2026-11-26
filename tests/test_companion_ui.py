from __future__ import annotations

import re
from pathlib import Path

import pytest

from familiar_brain import indexing
from familiar_brain.app import create_app
from familiar_brain.retrieval import Store
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm_server import FakeLLM

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def ui_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAMILIAR_NO_ROBOT", "1")
    fake = FakeLLM()
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    monkeypatch.setattr(indexing, "open_store", lambda cfg: store)

    persona = tmp_path / "persona.yaml"
    persona.write_text(
        "name: Rocky\n"
        "aliases: [rocky, rocki, rockie]\n"
        "tts_speaker: Aiden\n"
        "manner: dry\n"
        "gestures:\n"
        "  attentive: play_emotion:attentive\n"
        "  amused: play_emotion:happy\n"
        "  alarmed: play_emotion:surprised\n"
        "  thoughtful: play_emotion:thinking\n"
        "  sad: play_emotion:sad\n"
        "  scared: play_emotion:scared\n"
        "  asleep: play_emotion:sleepy\n"
        "  awake: play_emotion:welcoming\n"
        "  cheer: play_emotion:success\n",
        encoding="utf-8",
    )
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "sessions").mkdir(exist_ok=True)
    (tmp_path / "players.txt").write_text("TJ\nElias\n", encoding="utf-8")

    cfg = tmp_path / "brain.toml"
    cfg.write_text(
        f"""
[server]
host = "127.0.0.1"
port = 0

[upstream]
base_url = "{fake.base_url}"
generate_slot = 0
classify_slot = 1
classify_timeout_s = 0.5

[gate]
follow_up_window_s = 8

[paths]
persona = "{persona.as_posix()}"
sessions_dir = "{(tmp_path / 'sessions').as_posix()}"
players = "{(tmp_path / 'players.txt').as_posix()}"

[knowledge]
db_path = "{(tmp_path / 'k.db').as_posix()}"
notes_dir = "{(tmp_path / 'notes').as_posix()}"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7

[speakers]
enabled = true
backend = "acoustic"
store_path = "{(tmp_path / 'speakers.json').as_posix()}"
""",
        encoding="utf-8",
    )

    app = create_app(cfg)
    app.config["TESTING"] = True
    try:
        yield app.test_client(), app, fake, tmp_path
    finally:
        app.extensions["close"]()
        fake.close()


def test_root_serves_zero_build_companion_ui_html(ui_env) -> None:
    client, _app, _fake, _tmp = ui_env
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.content_type
    html = resp.get_data(as_text=True)
    assert re.search(r'<meta[^>]+name=["\']viewport["\'][^>]+width=device-width', html, re.I)
    assert "Rocky" in html
    low = html.lower()
    for token in (
        "mute", "unmute", "speak", "mode", "deep", "note",
        "checkpoint", "start", "end", "purge", "reindex",
        "reconnect", "enroll", "transcript", "latency", "robot",
    ):
        assert token in low, f"Missing token {token!r} in index.html"


def test_static_css_and_js_served_cleanly_with_touch_targets_and_xss_safety(ui_env) -> None:
    client, _app, _fake, _tmp = ui_env

    css_resp = client.get("/static/styles.css")
    assert css_resp.status_code == 200
    assert "text/css" in css_resp.content_type
    css = css_resp.get_data(as_text=True)
    assert "--bg-obsidian" in css and "--gold" in css and "48px" in css

    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "javascript" in js_resp.content_type
    js = js_resp.get_data(as_text=True)
    assert "textContent" in js
    assert "innerHTML" not in js

    for endpoint in (
        "/api/state",
        "/api/mute",
        "/api/unmute",
        "/api/mute_and_stop_logging",
        "/api/speak_now",
        "/api/mode",
        "/api/deep_night",
        "/api/note",
        "/api/session/checkpoint",
        "/api/session/start",
        "/api/session/end",
        "/api/session/",
        "/api/sessions",
        "/api/reindex",
        "/api/robot/status",
        "/api/robot/reconnect",
        "/api/speakers",
        "/api/speakers/enroll",
    ):
        assert endpoint in js, f"Missing endpoint {endpoint!r} in app.js"

    html = client.get("/").get_data(as_text=True)
    actions = re.findall(r'data-action="([^"]+)"', html)
    assert len(actions) >= 14
    for act in actions:
        assert act in js, f"HTML data-action {act!r} not handled in app.js"


def test_static_assets_contain_zero_external_cdns_or_urls(ui_env) -> None:
    client, _app, _fake, _tmp = ui_env
    for path in ("/", "/static/styles.css", "/static/app.js"):
        body = client.get(path).get_data(as_text=True).lower()
        for bad in ("http://", "https://", "//cdn", "unpkg", "jsdelivr", "cdnjs", "fonts.googleapis"):
            assert bad not in body, f"Forbidden external reference {bad!r} found in {path}"


def test_missing_static_and_traversal_blocked(ui_env) -> None:
    client, _app, _fake, _tmp = ui_env
    assert client.get("/static/does_not_exist.js").status_code == 404
    trav = client.get("/static/../../config/brain.toml")
    assert trav.status_code in (400, 403, 404)
    assert "[upstream]" not in trav.get_data(as_text=True)


def test_gitignore_tracks_familiar_brain_static() -> None:
    gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    entries = [ln.strip() for ln in gi.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert "familiar_brain/static/" not in entries
    assert "familiar_brain/static" not in entries


def test_robot_status_and_reconnect_respect_familiar_no_robot(ui_env, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _app, _fake, _tmp = ui_env
    from scripts import robot_connect

    def forbid_probe(*args, **kwargs):
        raise AssertionError("Physical robot probe must not run when FAMILIAR_NO_ROBOT=1")

    monkeypatch.setattr(robot_connect, "check_robot_status", forbid_probe)
    monkeypatch.setattr(robot_connect, "connect_and_configure", forbid_probe)

    status_resp = client.get("/api/robot/status")
    assert status_resp.status_code == 200
    st_data = status_resp.get_json()
    assert st_data["disabled"] is True
    assert st_data["backend_connected"] is False
    assert {"backend_connected", "host", "last_checked_at", "detail", "disabled"} <= set(st_data.keys())

    for payload in ({}, {"host": "10.0.0.100"}, {"host": "reachy-mini.local"}):
        rec_resp = client.post("/api/robot/reconnect", json=payload)
        assert rec_resp.status_code == 503
        rec_data = rec_resp.get_json()
        assert rec_data["ok"] is False
        assert rec_data["backend_connected"] is False
        assert rec_data["disabled"] is True
        assert "FAMILIAR_NO_ROBOT" in rec_data["detail"]


def test_robot_reconnect_loopback_success_and_failure_updates_api_state(ui_env, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _app, _fake, _tmp = ui_env
    from scripts import robot_connect

    calls: list[dict] = []

    def fake_connect(host, **kwargs):
        calls.append({"host": host, **kwargs})
        if kwargs.get("daemon_port") == 9999:
            return {
                "ok": False,
                "backend_connected": False,
                "host": host,
                "error": "daemon unreachable at 127.0.0.1:9999",
            }
        return {
            "ok": True,
            "backend_connected": True,
            "host": host,
            "error": None,
        }

    monkeypatch.setattr(robot_connect, "connect_and_configure", fake_connect)

    ok_resp = client.post(
        "/api/robot/reconnect",
        json={"host": "127.0.0.1", "daemon_port": 18000, "rpc_port": 17860, "host_ip": "127.0.0.1", "port": 8765},
    )
    assert ok_resp.status_code == 200
    ok_data = ok_resp.get_json()
    assert ok_data["ok"] is True
    assert ok_data["backend_connected"] is True
    assert ok_data["host"] == "127.0.0.1"
    assert calls[0]["fallback_host"] is None
    assert calls[0]["allow_fallback"] is False

    state = client.get("/api/state").get_json()
    assert state["robot"]["backend_connected"] is True
    assert state["robot"]["host"] == "127.0.0.1"

    fail_resp = client.post(
        "/api/robot/reconnect",
        json={"host": "127.0.0.1", "daemon_port": 9999, "rpc_port": 9998},
    )
    assert fail_resp.status_code == 503
    fail_data = fail_resp.get_json()
    assert fail_data["ok"] is False
    assert fail_data["backend_connected"] is False

    state_after = client.get("/api/state").get_json()
    assert state_after["robot"]["backend_connected"] is False


def test_robot_extension_hooks_and_bad_inputs(ui_env) -> None:
    client, app, _fake, _tmp = ui_env

    app.extensions["robot_probe"] = lambda: {
        "backend_connected": True,
        "host": "127.0.0.1",
        "detail": "fake probe ok",
        "disabled": False,
    }
    r_status = client.get("/api/robot/status")
    assert r_status.status_code == 200
    assert r_status.get_json()["backend_connected"] is True
    assert client.get("/api/state").get_json()["robot"]["detail"] == "fake probe ok"

    app.extensions["robot_reconnect"] = lambda body: {
        "ok": True,
        "backend_connected": True,
        "host": body.get("host") or "127.0.0.1",
        "detail": "hook connected",
    }
    r_rec = client.post("/api/robot/reconnect", json={"host": "127.0.0.1"})
    assert r_rec.status_code == 200
    assert r_rec.get_json()["detail"] == "hook connected"

    del app.extensions["robot_probe"]
    del app.extensions["robot_reconnect"]
    assert client.post("/api/robot/reconnect", json=["not", "a", "dict"]).status_code == 400
    assert client.post("/api/robot/reconnect", json={"host": "127.0.0.1", "daemon_port": "bad"}).status_code == 400
    assert client.get("/api/robot/status?host=127.0.0.1&daemon_port=bad").status_code == 400


def test_companion_ui_control_endpoints_end_to_end(ui_env) -> None:
    client, _app, fake, _tmp = ui_env

    sid = client.post("/api/session/start", json={}).get_json()["session"]
    assert sid

    assert client.post("/api/mute", json={}).status_code == 200
    assert client.get("/api/state").get_json()["muted"] is True

    assert client.post("/api/unmute", json={}).status_code == 200
    assert client.get("/api/state").get_json()["muted"] is False

    assert client.post("/api/mode", json={"mode": "thinking"}).status_code == 200
    assert client.get("/api/state").get_json()["mode"] == "thinking"

    assert client.post("/api/deep_night", json={"asleep": True}).status_code == 200
    assert client.get("/api/state").get_json()["asleep"] is True
    assert client.post("/api/speak_now", json={}).status_code == 409

    assert client.post("/api/deep_night", json={"asleep": False}).status_code == 200
    assert client.post("/api/speak_now", json={}).status_code == 200

    note_resp = client.post("/api/note", json={"text": "Found a silver key in the crypt."})
    assert note_resp.status_code == 200
    assert client.post("/api/reindex", json={}).status_code == 200

    fake.chat_replies.extend([
        "## Scenes\n- Silver key found.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- Silver key\n## Open threads\n- none\n## Memorable lines\n- none\n",
        "# Memory\n- Silver key found in the crypt.\n",
    ])
    ckpt = client.post("/api/session/checkpoint", json={"force": True})
    assert ckpt.status_code == 200
    assert ckpt.get_json()["ok"] is True

    assert client.delete(f"/api/session/{sid}").status_code == 409
    assert client.post("/api/session/end", json={}).status_code == 200
    assert client.delete(f"/api/session/{sid}").status_code == 200


def test_session_purge_rejects_path_traversal_and_preserves_canary(ui_env) -> None:
    from familiar_brain.summariser import purge_session

    client, app, _fake, tmp_path = ui_env
    canary = tmp_path / "foo.jsonl"
    canary.write_text('{"text":"do not delete"}\n', encoding="utf-8")

    for bad_url in (
        r"/api/session/..\foo",
        "/api/session/..%5Cfoo",
        "/api/session/..",
        "/api/session/.",
        "/api/session/a%5Cb",
    ):
        resp = client.delete(bad_url)
        assert resp.status_code == 400, f"{bad_url} returned {resp.status_code}"
        assert canary.exists()

    # Valid nonexistent sid still returns 200 {"purged": False}
    ok_resp = client.delete("/api/session/nope")
    assert ok_resp.status_code == 200
    assert ok_resp.get_json() == {"purged": False}

    # Direct purge_session also rejects traversal
    for bad_sid in ("../foo", r"..\foo", "..", ".", "a/b"):
        with pytest.raises(ValueError):
            purge_session(tmp_path / "sessions", bad_sid, store=app.extensions["store"])
    assert canary.exists()



def _js_function_body(js: str, signature: str) -> str:
    start = js.index(signature)
    end = js.index("\n  }\n", start)
    return js[start:end]


def test_app_js_has_no_undeclared_state_refs() -> None:
    js = (REPO_ROOT / "familiar_brain" / "static" / "app.js").read_text(encoding="utf-8")
    # The Reconnect button read an undeclared `lastState` and threw before any request.
    assert "lastState" not in js
    # The SSE listener calls applyState, which must exist or every live update throws.
    assert re.search(r"function applyState\s*\(", js)
    body = _js_function_body(js, "function applyState")
    assert "latestState" in body and "renderStatus" in body


def test_app_js_poll_cannot_wedge_and_good_poll_clears_banner() -> None:
    js = (REPO_ROOT / "familiar_brain" / "static" / "app.js").read_text(encoding="utf-8")
    api_body = _js_function_body(js, "async function apiJson")
    # A hung fetch must abort so refreshState's finally clears pollInFlight.
    assert "AbortController" in api_body
    refresh_body = _js_function_body(js, "async function refreshState")
    assert re.search(r'apiJson\("/api/state",\s*\{\s*timeoutMs:\s*5000', refresh_body)
    assert "hideBanner" in refresh_body
    # Only the poll is capped: slow actions (official-app Reconnect, enroll) must not abort at 5 s.
    assert "5000" not in api_body
    assert "timeoutMs" in api_body


def test_companion_plus_handles_missing_familiar_routes() -> None:
    js = (REPO_ROOT / "familiar_brain" / "static" / "companion_plus.js").read_text(encoding="utf-8")
    body = _js_function_body(js, "async function pollRobotBootStatus")
    # Served from the laptop :8090 the robot-only routes 404; the 750 ms poll must stop.
    assert "404" in body
    assert "clearInterval" in body
    assert "Only available from the robot UI on port 8043" in js


def test_mute_stop_publishes_state(ui_env, monkeypatch: pytest.MonkeyPatch) -> None:
    client, app, _fake, _tmp = ui_env
    published: list[tuple[str, dict]] = []
    bus = app.extensions["event_bus"]
    monkeypatch.setattr(bus, "publish", lambda kind, payload: published.append((kind, payload)))

    resp = client.post("/api/mute_and_stop_logging")
    assert resp.status_code == 200
    states = [p for k, p in published if k == "state"]
    assert states, "mute and stop logging must push a state update to the UI"
    assert states[-1]["muted"] is True
