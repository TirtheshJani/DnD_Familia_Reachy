from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest
import requests

from scripts import robot_connect


class _FakeResp:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.ok = 200 <= status_code < 400

    def json(self) -> dict | None:
        return self._payload


class _FakeRpc:
    def __init__(self, *, statuses: list[dict], voices: list[str] | None = None, loop_unavailable_calls: int = 0) -> None:
        self.statuses = list(statuses)
        self.voices = list(voices if voices is not None else ["Aiden", "Cedric"])
        self.loop_unavailable_calls = loop_unavailable_calls
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def __enter__(self) -> _FakeRpc:
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True

    def call(self, method: str, params: dict | None = None, timeout: float = 30):
        self.calls.append((method, params or {}))
        if method == "conversation.status":
            if len(self.statuses) > 1:
                return self.statuses.pop(0)
            return self.statuses[0]
        if method == "backend.config":
            return {"message": "ok"}
        if method == "voices.list":
            return list(self.voices)
        if method == "personalities.save":
            return {"value": (params or {}).get("name", "familiar")}
        if method == "profile_tools.save":
            return {"message": "saved", "enabled_tools": (params or {}).get("enabled_tools", [])}
        if method == "personalities.apply":
            if self.loop_unavailable_calls > 0:
                self.loop_unavailable_calls -= 1
                raise robot_connect.RpcError(method, {"message": "loop not ready", "data": {"reason": "loop_unavailable"}})
            return {"status": "applied", "startup": "ok"}
        if method == "voices.apply":
            return {"status": "voice_applied"}
        if method == "personalities.list":
            return ["familiar"]
        if method == "profile_tools.get":
            return {"available_tools": [{"id": "move_head"}]}
        raise AssertionError(f"unexpected RPC method: {method}")


def test_resolve_robot_host_prefers_mdns_when_healthy() -> None:
    def fake_resolver(host: str, port: int):
        assert host == "reachy-mini.local"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.222", port))]

    def fake_get(url: str, timeout: float = 3.0):
        assert url == "http://reachy-mini.local:8000/api/apps/current-app-status"
        return _FakeResp(200, {"state": "running"})

    host = robot_connect.resolve_robot_host(resolver=fake_resolver, http_get=fake_get)
    assert host == "reachy-mini.local"


def test_resolve_robot_host_falls_back_to_ip_when_mdns_fails() -> None:
    def fake_resolver(host: str, port: int):
        raise socket.gaierror("mDNS unavailable in WSL2")

    def fake_get(url: str, timeout: float = 3.0):
        assert url == "http://192.0.2.10:8000/api/apps/current-app-status"
        return _FakeResp(200, {"state": "running"})

    host = robot_connect.resolve_robot_host(
        candidates=("reachy-mini.local", "192.0.2.10"),
        fallback_host="192.0.2.10",
        resolver=fake_resolver,
        http_get=fake_get,
    )
    assert host == "192.0.2.10"


def test_resolve_robot_host_raises_when_nothing_resolves() -> None:
    def bad_resolver(host: str, port: int):
        raise socket.gaierror("nope")

    with pytest.raises(ConnectionError):
        robot_connect.resolve_robot_host(
            candidates=("missing1.local", "missing2.local"),
            fallback_host=None,
            resolver=bad_resolver,
            http_get=lambda *a, **k: _FakeResp(500),
        )


def test_resolve_robot_host_explicit_preferred_never_falls_back_to_10_0_0_100() -> None:
    probed_hosts: list[str] = []
    resolved_names: list[str] = []

    def tracking_resolver(host: str, port: int):
        resolved_names.append(host)
        raise socket.gaierror(f"cannot resolve {host}")

    def tracking_get(url: str, timeout: float = 3.0):
        probed_hosts.append(url)
        if "10.0.0.100" in url:
            raise AssertionError(f"Hardware safety violation: probed 10.0.0.100 via {url}")
        raise requests.ConnectionError("unreachable")

    # 1. Explicit unresolvable hostname must raise ConnectionError without touching 10.0.0.100
    with pytest.raises(ConnectionError):
        robot_connect.resolve_robot_host(
            preferred_host="unreachable-test-host.invalid",
            resolver=tracking_resolver,
            http_get=tracking_get,
        )
    assert resolved_names == ["unreachable-test-host.invalid"]
    assert probed_hosts == []

    # 2. Explicit unreachable IP must never probe or return 10.0.0.100
    host = robot_connect.resolve_robot_host(
        preferred_host="127.0.0.1",
        resolver=tracking_resolver,
        http_get=tracking_get,
    )
    assert host == "127.0.0.1"
    assert probed_hosts == ["http://127.0.0.1:8000/api/apps/current-app-status"]


def test_resolve_robot_host_explicit_preferred_allows_fallback_only_when_enabled() -> None:
    def bad_resolver(host: str, port: int):
        raise socket.gaierror("nope")

    def fake_get(url: str, timeout: float = 3.0):
        assert url == "http://192.0.2.99:8000/api/apps/current-app-status"
        return _FakeResp(200, {"state": "running"})

    host = robot_connect.resolve_robot_host(
        preferred_host="missing.invalid",
        fallback_host="192.0.2.99",
        allow_fallback=True,
        resolver=bad_resolver,
        http_get=fake_get,
    )
    assert host == "192.0.2.99"


def test_check_robot_status_and_connect_explicit_host_never_touch_10_0_0_100() -> None:
    probed_urls: list[str] = []

    def safe_get(url: str, timeout: float = 5.0):
        probed_urls.append(url)
        if "10.0.0.100" in url:
            raise AssertionError(f"Hardware safety violation: probed 10.0.0.100 via {url}")
        raise requests.ConnectionError("connection refused")

    status = robot_connect.check_robot_status(
        "127.0.0.1",
        http_get=safe_get,
        rpc_factory=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not open RPC")),
    )
    assert status["ok"] is False
    assert status["backend_connected"] is False
    assert status["host"] == "127.0.0.1"
    assert all("10.0.0.100" not in u for u in probed_urls)

    probed_urls.clear()
    res = robot_connect.connect_and_configure(
        "missing-robot.invalid",
        resolver=lambda h, p: (_ for _ in ()).throw(socket.gaierror("nope")),
        http_get=safe_get,
        http_post=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not POST")),
        rpc_factory=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not open RPC")),
        sleep=lambda _s: None,
    )
    assert res["ok"] is False
    assert res["backend_connected"] is False
    assert res["host"] == "missing-robot.invalid"
    assert probed_urls == []


def test_main_explicit_robot_without_fallback_never_sets_fallback_host(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict] = []

    def spy_connect(host, **kwargs):
        captured.append({"host": host, **kwargs})
        return {"ok": True, "backend_connected": True}

    monkeypatch.setattr(robot_connect, "connect_and_configure", spy_connect)

    assert robot_connect.main(["--robot", "127.0.0.1"]) == 0
    assert captured[-1]["host"] == "127.0.0.1"
    assert captured[-1]["fallback_host"] is None
    assert captured[-1]["allow_fallback"] is False

    assert robot_connect.main(["--robot", "auto"]) == 0
    assert captured[-1]["host"] is None
    assert captured[-1]["fallback_host"] == "10.0.0.100"

    assert robot_connect.main(["--robot", "bad.invalid", "--fallback-host", "192.0.2.10"]) == 0
    assert captured[-1]["host"] == "bad.invalid"
    assert captured[-1]["fallback_host"] == "192.0.2.10"
    assert captured[-1]["allow_fallback"] is True


def test_main_rejects_empty_robot_flag(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("connect_and_configure must not be called for empty --robot")

    monkeypatch.setattr(robot_connect, "connect_and_configure", fail_if_called)

    for raw in ("", "   "):
        with pytest.raises(SystemExit) as exc:
            robot_connect.main(["--robot", raw])
        assert exc.value.code == 2
        assert "--robot must not be empty" in capsys.readouterr().err


def test_resolve_robot_host_empty_preferred_acts_as_auto() -> None:
    def fake_resolver(host: str, port: int):
        raise socket.gaierror(f"cannot resolve {host}")

    def fake_get(url: str, timeout: float = 3.0):
        assert url == "http://192.0.2.44:8000/api/apps/current-app-status"
        return _FakeResp(200, {"state": "running"})

    for raw in ("", "   "):
        host = robot_connect.resolve_robot_host(
            candidates=("missing.local", "192.0.2.44"),
            preferred_host=raw,
            fallback_host="192.0.2.44",
            resolver=fake_resolver,
            http_get=fake_get,
        )
        assert host == "192.0.2.44"


def test_check_robot_status_reports_connected_and_disconnected() -> None:
    fake_rpc = _FakeRpc(statuses=[{"backend_connected": True, "mode": "local"}])

    def fake_get(url: str, timeout: float = 5.0):
        return _FakeResp(200, {"info": {"name": robot_connect.APP}, "state": "running"})

    res = robot_connect.check_robot_status(
        "192.0.2.10",
        http_get=fake_get,
        rpc_factory=lambda h, p, timeout=5.0: fake_rpc,
    )
    assert res["ok"] is True
    assert res["backend_connected"] is True
    assert res["app_running"] is True
    assert fake_rpc.closed is True

    # App not running -> ok False, no RPC call needed
    res_stopped = robot_connect.check_robot_status(
        "192.0.2.10",
        http_get=lambda u, timeout=5.0: _FakeResp(200, {"info": None, "state": "stopped"}),
        rpc_factory=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not open RPC")),
    )
    assert res_stopped["ok"] is False
    assert res_stopped["app_running"] is False
    assert res_stopped["backend_connected"] is False


def test_connect_and_configure_retries_transient_daemon_and_starts_app() -> None:
    get_responses = iter([
        requests.ConnectionError("daemon booting"),
        _FakeResp(503, None),
        _FakeResp(200, {"state": "starting"}),
        _FakeResp(200, {"info": None, "state": "stopped"}),
    ])
    post_responses = iter([
        _FakeResp(503, None, text="busy"),
        _FakeResp(200, {"ok": True}),
    ])
    rpc_attempts = {"n": 0}
    fake_rpc = _FakeRpc(
        statuses=[
            {"backend_connected": False},  # initial connect_when_ready
            {"backend_connected": False},  # poll 1 after apply
            {"backend_connected": True},   # poll 2 after apply
            {"backend_connected": True},   # post voices.apply status
        ],
        loop_unavailable_calls=1,
    )

    def fake_get(url: str, timeout: float = 5.0):
        item = next(get_responses)
        if isinstance(item, Exception):
            raise item
        return item

    def fake_post(url: str, timeout: float = 30.0):
        return next(post_responses)

    def fake_rpc_factory(host: str, port: int, timeout: float = 10.0):
        assert host == "192.0.2.10"
        assert port == 7860
        rpc_attempts["n"] += 1
        if rpc_attempts["n"] == 1:
            raise OSError("rpc still starting")
        return fake_rpc

    res = robot_connect.connect_and_configure(
        "192.0.2.10",
        host_ip="192.0.2.50",
        port=8765,
        voice="Aiden",
        daemon_port=8000,
        rpc_port=7860,
        http_get=fake_get,
        http_post=fake_post,
        rpc_factory=fake_rpc_factory,
        sleep=lambda _s: None,
    )
    assert res["ok"] is True
    assert res["backend_connected"] is True
    assert res["voice"] == "Aiden"
    methods = [m for m, _ in fake_rpc.calls]
    assert "backend.config" in methods
    assert "personalities.save" in methods
    assert "profile_tools.save" in methods
    assert "personalities.apply" in methods
    assert "voices.apply" in methods


def test_connect_and_configure_unconditionally_checks_backend_connected_without_voice() -> None:
    fake_rpc = _FakeRpc(
        statuses=[
            {"backend_connected": False},  # connect_when_ready
            {"backend_connected": False},  # poll 1
            {"backend_connected": False},  # poll timeout -> still False!
        ],
        voices=[],  # requested voice absent, so voices.apply will be skipped
    )
    session = SimpleNamespace(
        get=lambda u, timeout=5.0: _FakeResp(200, {"info": {"name": robot_connect.APP}, "state": "running"}),
        post=lambda u, timeout=30.0: _FakeResp(200, {}),
    )
    res = robot_connect.connect_and_configure(
        "192.0.2.10",
        host_ip="192.0.2.50",
        voice="Aiden",
        verify_timeout_s=0.0,
        session=session,
        rpc_factory=lambda h, p, timeout=10.0: fake_rpc,
        sleep=lambda _s: None,
    )
    assert res["ok"] is False
    assert res["backend_connected"] is False
    assert "backend_connected" in (res["error"] or "")
    assert "voices.apply" not in [m for m, _ in fake_rpc.calls]


def test_main_exits_nonzero_when_backend_not_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        robot_connect,
        "connect_and_configure",
        lambda *a, **k: {"ok": False, "backend_connected": False, "error": "still disconnected"},
    )
    with pytest.raises(SystemExit) as exc:
        robot_connect.main(["--robot", "192.0.2.10", "--host-ip", "192.0.2.50"])
    assert "still disconnected" in str(exc.value)


def test_dry_run_does_not_start_stopped_app() -> None:
    session = SimpleNamespace(
        get=lambda u, timeout=5.0: _FakeResp(200, {"info": None, "state": "stopped"}),
        post=lambda *a, **k: (_ for _ in ()).throw(AssertionError("dry-run must not POST start-app")),
    )
    res = robot_connect.connect_and_configure(
        "192.0.2.10",
        host_ip="192.0.2.50",
        dry_run_mode=True,
        session=session,
        rpc_factory=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not connect RPC")),
    )
    assert res["ok"] is False
    assert "dry-run" in (res["error"] or "")


def _running_session(info: dict) -> SimpleNamespace:
    return SimpleNamespace(
        get=lambda u, timeout=5.0: _FakeResp(200, {"info": info, "state": "running"}),
        post=lambda u, timeout=30.0: _FakeResp(200, {}),
    )


def _connect(fake_rpc: _FakeRpc, session: SimpleNamespace) -> dict:
    return robot_connect.connect_and_configure(
        "192.0.2.10",
        host_ip="192.0.2.50",
        port=8765,
        voice="Aiden",
        session=session,
        rpc_factory=lambda h, p, timeout=10.0: fake_rpc,
        sleep=lambda _s: None,
    )


def test_warns_on_old_app_version(caplog: pytest.LogCaptureFixture) -> None:
    fake_rpc = _FakeRpc(statuses=[{"backend_connected": True}])
    with caplog.at_level("WARNING"):
        res = _connect(fake_rpc, _running_session({"name": robot_connect.APP, "version": "1.0.1"}))
    assert res["ok"] is True
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("1.0.1" in m and "1.2.0" in m for m in warnings), warnings


def test_no_warning_on_current_version(caplog: pytest.LogCaptureFixture) -> None:
    assert robot_connect.parse_app_version({"info": {"version": "1.2.0"}}) == (1, 2, 0)
    assert robot_connect.parse_app_version({"info": None}) is None
    fake_rpc = _FakeRpc(statuses=[{"backend_connected": True}])
    with caplog.at_level("WARNING"):
        res = _connect(fake_rpc, _running_session({"name": robot_connect.APP, "version": "1.2.0"}))
    assert res["ok"] is True
    assert [r for r in caplog.records if r.levelno >= 30] == []


def test_configure_skips_backend_config_when_unchanged() -> None:
    fake_rpc = _FakeRpc(statuses=[{"backend_connected": True, "host": "192.0.2.50", "port": 8765}])
    res = _connect(fake_rpc, _running_session({"name": robot_connect.APP, "version": "1.2.0"}))
    assert res["ok"] is True
    methods = [m for m, _ in fake_rpc.calls]
    assert "backend.config" not in methods
    assert "personalities.apply" in methods


def test_skipped_backend_config_still_restarts_backend_once() -> None:
    # Reconnect on a stuck app that still reports connected must restart its backend exactly once.
    fake_rpc = _FakeRpc(statuses=[{"backend_connected": True, "host": "192.0.2.50", "port": 8765}])
    _connect(fake_rpc, _running_session({"name": robot_connect.APP, "version": "1.2.0"}))
    methods = [m for m, _ in fake_rpc.calls]
    applies = [p for m, p in fake_rpc.calls if m == "personalities.apply"]
    assert "backend.config" not in methods
    assert applies and applies[0].get("force") is True


def test_version_check_ignores_another_running_app(caplog: pytest.LogCaptureFixture) -> None:
    fake_rpc = _FakeRpc(statuses=[{"backend_connected": True}])
    with caplog.at_level("WARNING"):
        _connect(fake_rpc, _running_session({"name": "hand_tracker_v2", "version": "1.0.1"}))
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert not any("1.0.1" in m for m in warnings), warnings


def test_apply_is_not_forced() -> None:
    fake_rpc = _FakeRpc(statuses=[{"backend_connected": True}])
    _connect(fake_rpc, _running_session({"name": robot_connect.APP, "version": "1.2.0"}))
    applies = [p for m, p in fake_rpc.calls if m == "personalities.apply"]
    assert applies and all("force" not in p for p in applies)
    assert applies[0]["persist"] is True
