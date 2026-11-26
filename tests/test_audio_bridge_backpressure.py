"""Native app audio bridge: the mic sender must survive send stalls, and if it dies the session must reconnect."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "reachy_familiar") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "reachy_familiar"))

from reachy_familiar import audio_bridge  # noqa: E402
from reachy_familiar.audio_bridge import RealtimeAudioBridge, SimpleWebSocket  # noqa: E402

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class _FakeCreature:
    def set_wobbling(self, on: bool) -> None:
        pass

    def play_emotion(self, *args, **kwargs) -> None:
        pass


class _FakeMedia:
    """Stereo float32 mic frames at the Realtime rate, one every `period` seconds."""

    def __init__(self, frames: int = 4096, period: float = 0.01) -> None:
        self.frames = frames
        self.period = period

    def get_audio_sample(self):
        time.sleep(self.period)
        return np.full((self.frames, 2), 0.01, dtype=np.float32)


class _FakeReachy:
    def __init__(self, media: _FakeMedia) -> None:
        self.media = media


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed")
        buf.extend(chunk)
    return bytes(buf)


def _handshake(conn: socket.socket) -> None:
    buf = bytearray()
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            raise ConnectionError("client closed during handshake")
        buf.extend(chunk)
    head = bytes(buf).decode("latin-1")
    key = ""
    for line in head.split("\r\n")[1:]:
        name, _, value = line.partition(":")
        if name.strip().lower() == "sec-websocket-key":
            key = value.strip()
    accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()).decode("ascii")
    conn.sendall(
        (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode("ascii")
    )


def _read_client_frame(conn: socket.socket) -> tuple[int, bytes]:
    b0, b1 = _recv_exact(conn, 2)
    n = b1 & 0x7F
    if n == 126:
        (n,) = struct.unpack(">H", _recv_exact(conn, 2))
    elif n == 127:
        (n,) = struct.unpack(">Q", _recv_exact(conn, 8))
    mask = _recv_exact(conn, 4) if (b1 & 0x80) else b""
    payload = _recv_exact(conn, n)
    if mask:
        payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
    return b0 & 0x0F, payload


def _server_frame(opcode: int, payload: bytes) -> bytes:
    assert len(payload) < 126
    return struct.pack(">BB", 0x80 | opcode, len(payload)) + payload


class _FakeWSServer:
    """Accepts WebSocket clients on 127.0.0.1 and runs `handler(conn, index)` per connection in a thread."""

    def __init__(self, handler, rcvbuf: int | None = None) -> None:
        self.handler = handler
        self.lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if rcvbuf is not None:
            # Set before listen() so accepted sockets inherit the small receive window.
            self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        self.lsock.bind(("127.0.0.1", 0))
        self.lsock.listen(4)
        self.lsock.settimeout(0.2)
        self.port = self.lsock.getsockname()[1]
        self.stop = threading.Event()
        self.conns: list[socket.socket] = []
        self.threads: list[threading.Thread] = []
        self._accept = threading.Thread(target=self._accept_loop, daemon=True, name="fake-ws-accept")
        self._accept.start()

    def _accept_loop(self) -> None:
        while not self.stop.is_set():
            try:
                conn, _ = self.lsock.accept()
            except (socket.timeout, OSError):
                continue
            conn.settimeout(5.0)
            index = len(self.conns)
            self.conns.append(conn)
            t = threading.Thread(target=self._serve, args=(conn, index), daemon=True, name=f"fake-ws-{index}")
            self.threads.append(t)
            t.start()

    def _serve(self, conn: socket.socket, index: int) -> None:
        try:
            _handshake(conn)
            self.handler(conn, index)
        except (OSError, ConnectionError):
            pass

    def close(self) -> None:
        self.stop.set()
        for conn in list(self.conns):
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        try:
            self.lsock.close()
        except OSError:
            pass
        self._accept.join(timeout=2.0)
        for t in self.threads:
            t.join(timeout=2.0)


def _mic_thread_alive() -> bool:
    return any(t.name == "reachy-mic-tx" and t.is_alive() for t in threading.enumerate())


def test_mic_thread_survives_or_reconnects_after_send_stall():
    stall_end = threading.Event()
    stall_end_at = [0.0]
    appends_after_stall: dict[int, int] = {}

    def handler(conn: socket.socket, index: int) -> None:
        if index == 0:
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.0:
                _read_client_frame(conn)
            # Stop reading: the client's send buffer fills and sendall blocks.
            time.sleep(4.0)
            stall_end_at[0] = time.monotonic()
            stall_end.set()
        appends_after_stall.setdefault(index, 0)
        last_ping = 0.0
        while True:
            now = time.monotonic()
            if now - last_ping >= 1.0:
                conn.sendall(_server_frame(0x9, b"hb"))
                last_ping = now
            op, payload = _read_client_frame(conn)
            # Skip the first second after the stall so the kernel backlog queued before it does not count.
            if op == 0x1 and stall_end.is_set() and time.monotonic() - stall_end_at[0] >= 1.0:
                try:
                    if json.loads(payload).get("type") == "input_audio_buffer.append":
                        appends_after_stall[index] += 1
                except ValueError:
                    pass

    server = _FakeWSServer(handler, rcvbuf=8192)
    stop_event = threading.Event()
    bridge = RealtimeAudioBridge(
        host="127.0.0.1",
        port=server.port,
        reachy_mini=_FakeReachy(_FakeMedia(frames=4096, period=0.01)),
        creature=_FakeCreature(),
    )
    runner = threading.Thread(target=bridge.run_loop, args=(stop_event,), daemon=True, name="bridge-run-loop")
    try:
        runner.start()
        assert stall_end.wait(timeout=8.0), "fake server never finished its stall"
        time.sleep(4.0)
        mic_alive = _mic_thread_alive()
        current = len(server.conns) - 1
        assert mic_alive or current >= 1, "mic thread is dead and the bridge never reconnected"
        assert appends_after_stall.get(current, 0) >= 1, (
            f"no input_audio_buffer.append read after the stall on connection {current}: {appends_after_stall}"
        )
        if not mic_alive:
            assert bridge.connected is False, "bridge reports connected while no mic thread is sending"
    finally:
        stop_event.set()
        server.close()
        runner.join(timeout=5.0)
    assert not runner.is_alive()


def test_partial_frame_does_not_tear_down_session():
    text = "hello rocky, this frame arrives in two pieces"
    frame = _server_frame(0x1, text.encode("utf-8"))

    def handler(conn: socket.socket, index: int) -> None:
        conn.sendall(frame[:3])
        time.sleep(0.8)
        conn.sendall(frame[3:])
        time.sleep(3.0)

    server = _FakeWSServer(handler)
    ws = None
    try:
        ws = SimpleWebSocket("127.0.0.1", server.port, "/v1/realtime", timeout=10.0)
        got = None
        deadline = time.monotonic() + 4.0
        while got is None and time.monotonic() < deadline:
            got = ws.recv_text(timeout=0.5)
        assert got == text
    finally:
        if ws is not None:
            ws.close()
        server.close()


def test_mic_send_failure_triggers_reconnect(monkeypatch, caplog):
    class _FailingWS:
        instances: list["_FailingWS"] = []

        def __init__(self, host, port, path="/v1/realtime", timeout=10.0) -> None:
            self.calls = 0
            self.closed = threading.Event()
            _FailingWS.instances.append(self)

        def send_json(self, obj) -> None:
            self.calls += 1
            if self.calls == 5:
                raise TimeoutError("timed out")

        def recv_text(self, timeout: float = 1.0):
            time.sleep(min(timeout, 0.05))
            return None

        def close(self) -> None:
            self.closed.set()

    monkeypatch.setattr(audio_bridge, "SimpleWebSocket", _FailingWS)
    caplog.set_level(logging.WARNING, logger="reachy_familiar.audio_bridge")

    stop_event = threading.Event()
    bridge = RealtimeAudioBridge(
        host="127.0.0.1",
        port=1,
        reachy_mini=_FakeReachy(_FakeMedia(frames=320, period=0.01)),
        creature=_FakeCreature(),
    )
    outcome: dict[str, BaseException | None] = {"exc": None}

    def run() -> None:
        try:
            bridge._run_session(stop_event)
        except BaseException as exc:  # noqa: BLE001
            outcome["exc"] = exc

    runner = threading.Thread(target=run, daemon=True, name="bridge-session")
    try:
        runner.start()
        runner.join(timeout=2.0)
        assert not runner.is_alive(), "_run_session kept running after the mic sender died"
    finally:
        stop_event.set()
        runner.join(timeout=3.0)
    assert isinstance(outcome["exc"], ConnectionError)
    assert _FailingWS.instances[0].closed.is_set(), "mic sender did not close the WebSocket"
    assert not bridge.mic_alive
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "mic" in r.getMessage().lower()]
    assert warnings, "no WARNING mentioning the mic was logged"


class _StallingMedia:
    """Mic frames for `good_for` seconds, then None. With `recover`, a stop_recording then
    start_recording brings frames back. With `stop_raises`, stop_recording raises."""

    def __init__(self, good_for: float, recover: bool, stop_raises: bool = False) -> None:
        self.t0 = time.monotonic()
        self.good_for = good_for
        self.recover = recover
        self.stop_raises = stop_raises
        self.stopped = False
        self.recovered = False
        self.calls: list[tuple[str, float]] = []

    def get_audio_sample(self):
        time.sleep(0.01)
        if self.recovered or time.monotonic() - self.t0 < self.good_for:
            return np.full((320, 2), 0.01, np.float32)
        return None

    def stop_recording(self) -> None:
        self.calls.append(("stop_recording", time.monotonic()))
        self.stopped = True
        if self.stop_raises:
            raise RuntimeError("alsa device gone")

    def start_recording(self) -> None:
        self.calls.append(("start_recording", time.monotonic()))
        if self.recover and self.stopped:
            self.recovered = True

    def start_playing(self) -> None:
        self.calls.append(("start_playing", time.monotonic()))


class _RecordingWS:
    def __init__(self) -> None:
        self.sent: list[tuple[str, float]] = []
        self.closed = threading.Event()

    def send_json(self, obj) -> None:
        self.sent.append((obj.get("type"), time.monotonic()))

    def close(self) -> None:
        self.closed.set()


def _run_mic_loop(bridge: RealtimeAudioBridge, ws: _RecordingWS, seconds: float) -> threading.Thread:
    stop_event = threading.Event()
    mic_stop = threading.Event()
    t = threading.Thread(target=bridge._mic_loop, args=(ws, stop_event, mic_stop), daemon=True, name="test-mic")
    try:
        t.start()
        ws.closed.wait(timeout=seconds)
    finally:
        mic_stop.set()
        stop_event.set()
        t.join(timeout=2.0)
    assert not t.is_alive(), "_mic_loop did not stop"
    return t


def test_mic_loop_restarts_dead_capture(caplog):
    caplog.set_level(logging.WARNING, logger="reachy_familiar.audio_bridge")
    media = _StallingMedia(good_for=0.5, recover=True)
    bridge = RealtimeAudioBridge(
        host="127.0.0.1", port=1, reachy_mini=_FakeReachy(media), creature=_FakeCreature()
    )
    ws = _RecordingWS()
    stall_at = media.t0 + 0.5
    stop_event = threading.Event()
    mic_stop = threading.Event()
    t = threading.Thread(target=bridge._mic_loop, args=(ws, stop_event, mic_stop), daemon=True, name="test-mic")
    try:
        t.start()
        time.sleep(3.5)
    finally:
        mic_stop.set()
        stop_event.set()
        t.join(timeout=2.0)
    assert not t.is_alive(), "_mic_loop did not stop"

    names = [n for n, _ in media.calls]
    assert "stop_recording" in names and "start_recording" in names, f"no capture restart: {media.calls}"
    stop_t = next(ts for n, ts in media.calls if n == "stop_recording")
    start_t = next(ts for n, ts in media.calls if n == "start_recording")
    assert stop_t <= start_t
    assert start_t - stall_at <= 2.5, f"restart came {start_t - stall_at:.2f}s after the stall"
    assert any(
        r.levelno >= logging.WARNING and "stalled" in r.getMessage() for r in caplog.records
    ), "no WARNING containing 'stalled'"
    resumed = [ts for typ, ts in ws.sent if typ == "input_audio_buffer.append" and ts > start_t]
    assert resumed, "no input_audio_buffer.append after the restart"
    assert not ws.closed.is_set(), "a successful restart must not force a reconnect"
    assert bridge.mic_stalled is False


def test_capture_dead_after_two_restarts_forces_reconnect(caplog):
    caplog.set_level(logging.WARNING, logger="reachy_familiar.audio_bridge")
    # Never recovers, and stop_recording itself raises: the restart must log it and carry on.
    media = _StallingMedia(good_for=0.0, recover=False, stop_raises=True)
    bridge = RealtimeAudioBridge(
        host="127.0.0.1", port=1, reachy_mini=_FakeReachy(media), creature=_FakeCreature()
    )
    ws = _RecordingWS()
    _run_mic_loop(bridge, ws, seconds=6.0)

    assert ws.closed.is_set(), "dead capture never closed the WebSocket"
    assert bridge.mic_stalled is True
    starts = [n for n, _ in media.calls if n == "start_recording"]
    assert len(starts) == 2, f"expected 2 restarts before giving up: {media.calls}"
    assert any(
        r.levelno >= logging.WARNING and "stop_recording" in r.getMessage() for r in caplog.records
    ), "stop_recording failure was not logged at WARNING"
