from __future__ import annotations

import io
import json
import logging
import socket
import sys
import threading
import time
from pathlib import Path

import requests

# Ensure reachy_familiar package is importable directly from repo root during tests.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "reachy_familiar") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "reachy_familiar"))

import familiar_brain.wake_daemon as wake_daemon  # noqa: E402
from familiar_brain.wake_daemon import DISCOVER_MAGIC, StackManager, WakeDaemonServer  # noqa: E402
from reachy_familiar.audio_bridge import apply_speaker_boost  # noqa: E402
from reachy_familiar.creature import ALL_VALID_MOVES, CreatureController, EMOTION_LIBRARY, resolve_emotion_move  # noqa: E402
from reachy_familiar.web import CompanionProxyServer  # noqa: E402


class _FakeProc:
    def __init__(self, lines: list[str]) -> None:
        self.stdout = io.StringIO("\n".join(lines) + "\n")
        self._ended = threading.Event()
        self._rc: int | None = None

    def poll(self) -> int | None:
        return self._rc

    def wait(self, timeout: float | None = None) -> int:
        self._ended.wait(timeout=timeout)
        return 0 if self._rc is None else self._rc

    def terminate(self) -> None:
        self._rc = 0
        self._ended.set()


def test_wake_daemon_http_and_udp_discovery(tmp_path: Path):
    active_proc: list[_FakeProc] = []
    stopped_called = []

    def spawn() -> _FakeProc:
        p = _FakeProc([
            "llama-server ready",
            "familiar-brain ready",
            "speech-to-speech ready",
            "audio-tap ready",
            "Session auto-started",
        ])
        active_proc.append(p)
        return p

    def stop(proc) -> None:
        stopped_called.append(True)
        if proc is not None:
            proc.terminate()

    mgr = StackManager(
        repo_root=tmp_path,
        spawn_fn=spawn,
        stop_fn=stop,
        probe_fn=lambda: False,
    )
    srv = WakeDaemonServer(mgr, host="127.0.0.1", wake_port=0, discover_port=0)
    srv.start()
    try:
        # Wait briefly for UDP socket to bind its ephemeral port
        for _ in range(30):
            if srv.discover_port != 0:
                break
            time.sleep(0.05)

        # 1. Status starts sleeping
        r0 = requests.get(f"http://127.0.0.1:{srv.wake_port}/status", timeout=3).json()
        assert r0["state"] == "sleeping"

        # 2. UDP discovery answers with wake/brain/ws ports
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2.0)
            s.sendto(DISCOVER_MAGIC, ("127.0.0.1", srv.discover_port))
            data, _ = s.recvfrom(2048)
            disc = json.loads(data.decode("utf-8"))
            assert disc["service"] == "reachy_familiar_wake"
            assert disc["wake_port"] == srv.wake_port

        # 3. POST /wake transitions through boot lines to ready
        requests.post(f"http://127.0.0.1:{srv.wake_port}/wake", timeout=3)
        for _ in range(40):
            st = requests.get(f"http://127.0.0.1:{srv.wake_port}/status", timeout=3).json()
            if st["state"] == "ready":
                break
            time.sleep(0.05)
        assert st["state"] == "ready"
        assert st["stage"] == "ready"

        # 4. POST /sleep stops cleanly and returns to sleeping
        r_sleep = requests.post(f"http://127.0.0.1:{srv.wake_port}/sleep", timeout=3).json()
        assert r_sleep["state"] == "sleeping"
        assert stopped_called == [True]
    finally:
        srv.stop()


def test_wake_daemon_logs_to_file(tmp_path: Path):
    """Under pythonw.exe there is no stderr, so the daemon's own log must go to a file."""
    mgr = StackManager(repo_root=tmp_path, probe_fn=lambda: False)
    root = logging.getLogger()
    old_level = root.level
    handler = wake_daemon.setup_logging(mgr.repo_root)
    try:
        wake_daemon.log.info("wake daemon file log check")
        handler.flush()
        text = (tmp_path / "data" / "logs" / "wake_daemon.log").read_text(encoding="utf-8")
        assert "wake daemon file log check" in text
        assert handler.maxBytes == 5 * 1024 * 1024 and handler.backupCount == 3
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(old_level)


def test_wake_daemon_logs_stack_output_to_run_dir_stack_log(tmp_path: Path, monkeypatch):
    """The daemon names the run dir, hands it to start_all.sh as RUN_DIR, and appends the
    stack's stdout to that run dir's stack.log."""
    calls: list[list[str]] = []
    procs: list[_FakeProc] = []

    def fake_popen(cmd, **kwargs):
        calls.append(cmd)
        proc = _FakeProc(["familiar-brain ready", "Session auto-started"])
        procs.append(proc)
        return proc

    monkeypatch.setattr(wake_daemon.subprocess, "Popen", fake_popen)
    mgr = StackManager(repo_root=tmp_path, probe_fn=lambda: False, stop_fn=lambda proc: proc.terminate())
    mgr.wake()
    try:
        runs = list((tmp_path / "data" / "logs").iterdir())
        assert len(runs) == 1
        assert f"RUN_DIR='data/logs/{runs[0].name}' ./scripts/start_all.sh" in calls[0][-1]
        stack_log = runs[0] / "stack.log"
        deadline = time.monotonic() + 3
        while "Session auto-started" not in (stack_log.read_text(encoding="utf-8") if stack_log.exists() else ""):
            assert time.monotonic() < deadline, "stack.log never got the stack output"
            time.sleep(0.05)
        assert "familiar-brain ready" in stack_log.read_text(encoding="utf-8")
    finally:
        mgr.sleep()


def test_wake_daemon_watcher_keeps_the_stack_log_of_its_own_run(tmp_path: Path, monkeypatch):
    """A quick sleep/wake must not point the previous run's watcher at the next run's stack.log."""
    monkeypatch.setattr(wake_daemon.subprocess, "Popen", lambda cmd, **kw: _FakeProc(["first run line"]))
    started: list[threading.Thread] = []
    real_start = threading.Thread.start

    class _HeldThread(threading.Thread):
        def start(self) -> None:  # held until the next run's path is set
            started.append(self)

    monkeypatch.setattr(wake_daemon.threading, "Thread", _HeldThread)
    mgr = StackManager(repo_root=tmp_path, probe_fn=lambda: False, stop_fn=lambda proc: proc.terminate())
    mgr.wake()
    first_log = mgr._stack_log_path
    mgr._stack_log_path = tmp_path / "next_run_stack.log"
    real_start(started[0])
    started[0].join(timeout=3)
    try:
        assert "first run line" in first_log.read_text(encoding="utf-8")
        assert not (tmp_path / "next_run_stack.log").exists()
    finally:
        mgr.sleep()


def test_creature_controller_emotions_sleep_and_ambient_sway():
    # Every mapped intent resolves to a valid recorded move
    for intent in EMOTION_LIBRARY:
        move = resolve_emotion_move(intent)
        assert move in ALL_VALID_MOVES

    calls: list[tuple[str, str, object]] = []

    class FakeResp:
        def __init__(self, data=None):
            self.ok = True
            self._data = data or {}

        def json(self):
            return self._data

    class FakeHttp:
        def post(self, url, json=None, timeout=2.0):
            calls.append(("POST", url, json))
            return FakeResp()

        def get(self, url, timeout=1.0):
            calls.append(("GET", url, None))
            return FakeResp({"speech_detected": True, "angle": 1.2})

    ctrl = CreatureController(daemon_url="http://127.0.0.1:8000", session=FakeHttp())

    # Mute triggers true sleep pose (goto_sleep)
    act = ctrl.tick_ambient(ambient={"mood": "calm_music"}, behavior={"curiosity": 80, "expressiveness": 80},
                            muted=True, asleep=False, speaking=False, now=10.0)
    assert act == "sleep_pose"
    assert any("/api/move/play/goto_sleep" in u for _, u, _ in calls)

    # Unmute wakes cleanly
    act2 = ctrl.tick_ambient(ambient={"mood": "quiet"}, behavior={"curiosity": 80, "expressiveness": 80},
                             muted=False, asleep=False, speaking=False, now=12.0)
    assert act2 == "wake_pose"
    assert any("/api/move/play/wake_up" in u for _, u, _ in calls)

    # Calm music triggers gentle antenna sway
    act3 = ctrl.tick_ambient(ambient={"mood": "calm_music"}, behavior={"curiosity": 70, "expressiveness": 60},
                             muted=False, asleep=False, speaking=False, now=20.0)
    assert act3 == "music_sway"

    # Expressiveness=0 and curiosity=0 keeps motors quiet
    act4 = ctrl.tick_ambient(ambient={"mood": "combat_music"}, behavior={"curiosity": 0, "expressiveness": 0},
                             muted=False, asleep=False, speaking=False, now=60.0)
    assert act4 is None


def test_companion_proxy_shows_boot_screen_then_proxies_companion_ui(tmp_path: Path):
    # Boot a tiny fake upstream brain server
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class FakeBrain(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = b'{"ok": true, "mode": "familiar"}' if self.path == "/api/state" else b"<h1>Rocky Companion</h1>"
            ct = "application/json" if self.path == "/api/state" else "text/html"
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream = HTTPServer(("127.0.0.1", 0), FakeBrain)
    up_port = upstream.server_address[1]
    t = threading.Thread(target=upstream.serve_forever, daemon=True)
    t.start()

    boot = {"ready": False, "brain_ready": False, "voice_ready": False, "state": "waking", "stage": "llama", "detail": "Loading Gemma 4", "brain_url": f"http://127.0.0.1:{up_port}"}
    actions_called: list[str] = []
    proxy = CompanionProxyServer(
        host="127.0.0.1",
        port=0,
        status_provider=lambda: dict(boot),
        action_handlers={
            "speak_test": lambda: (actions_called.append("speak_test") or {"ok": True, "detail": "chime"}),
            "reconnect_audio": lambda: (actions_called.append("reconnect_audio") or {"ok": True, "detail": "reconnecting"}),
        },
    )
    proxy.start()
    try:
        r_boot = requests.get(f"http://127.0.0.1:{proxy.port}/", timeout=3)
        assert r_boot.status_code == 200
        assert "Waking Familiar" in r_boot.text

        # Query strings and browser routes while waking must still render HTML, never raw JSON.
        r_qs = requests.get(f"http://127.0.0.1:{proxy.port}/?from=desktop_app", timeout=3)
        assert r_qs.status_code == 200
        assert "text/html" in r_qs.headers.get("Content-Type", "")
        assert "Waking Familiar" in r_qs.text

        r_st = requests.get(f"http://127.0.0.1:{proxy.port}/_familiar/boot-status", timeout=3).json()
        assert r_st["ready"] is False and r_st["stage"] == "llama"

        # Companion UI opens as soon as brain_ready is True even while voice_ready is still warming up.
        boot["brain_ready"] = True
        r_ui = requests.get(f"http://127.0.0.1:{proxy.port}/", timeout=3)
        assert r_ui.status_code == 200
        assert "Rocky Companion" in r_ui.text

        r_api = requests.get(f"http://127.0.0.1:{proxy.port}/api/state", timeout=3).json()
        assert r_api["ok"] is True

        # Local Reachy Table Check actions work directly on :8043.
        r_sp = requests.post(f"http://127.0.0.1:{proxy.port}/_familiar/speak-test", json={}, timeout=3).json()
        r_rc = requests.post(f"http://127.0.0.1:{proxy.port}/_familiar/reconnect-audio", json={}, timeout=3).json()
        assert r_sp["ok"] is True and r_rc["ok"] is True
        assert actions_called == ["speak_test", "reconnect_audio"]
    finally:
        proxy.stop()
        upstream.shutdown()
        upstream.server_close()


def test_speaker_boost_applies_gain_with_soft_limit():
    import numpy as np

    quiet = np.array([0.1, -0.1, 0.25], dtype=np.float32)
    boosted = apply_speaker_boost(quiet, 6.0)
    assert float(np.max(np.abs(boosted))) > float(np.max(np.abs(quiet))) * 1.5
    loud = np.array([0.9, -0.95, 1.0], dtype=np.float32)
    limited = apply_speaker_boost(loud, 15.0)
    assert float(np.max(np.abs(limited))) <= 1.0
