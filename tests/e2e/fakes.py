"""Opaque-box E2E fake servers, RFC 6455 WebSocket client, and synthetic multi-speaker audio generators.

STRICT OPAQUE-BOX RULE: This module MUST NOT import any ``familiar_brain`` symbols.
All communication with the system under test uses HTTP, SSE, WebSocket, files, or CLI subprocesses.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import socket
import socketserver
import struct
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import requests

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ---------------------------------------------------------------------------
# Low-level RFC 6455 WebSocket helpers (pure stdlib)
# ---------------------------------------------------------------------------

def _ws_accept_key(client_key: str) -> str:
    return base64.b64encode(hashlib.sha1((client_key + WS_GUID).encode("ascii")).digest()).decode("ascii")


def _ws_encode_frame(opcode: int, payload: bytes, *, masked: bool = False) -> bytes:
    n = len(payload)
    mask_bit = 0x80 if masked else 0x00
    if n < 126:
        header = struct.pack(">BB", 0x80 | opcode, mask_bit | n)
    elif n < 65536:
        header = struct.pack(">BBH", 0x80 | opcode, mask_bit | 126, n)
    else:
        header = struct.pack(">BBQ", 0x80 | opcode, mask_bit | 127, n)
    if masked:
        mask = os.urandom(4)
        body = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        return header + mask + body
    return header + payload


def _recv_exact(sock: socket.socket, buf: bytearray, n: int) -> bytes:
    while len(buf) < n:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("WebSocket closed")
        buf.extend(chunk)
    out = bytes(buf[:n])
    del buf[:n]
    return out


def _ws_read_frame(sock: socket.socket, buf: bytearray) -> tuple[int, bytes]:
    parts: list[bytes] = []
    first_opcode: int | None = None
    while True:
        b0, b1 = _recv_exact(sock, buf, 2)
        opcode = b0 & 0x0F
        fin = bool(b0 & 0x80)
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F
        if length == 126:
            (length,) = struct.unpack(">H", _recv_exact(sock, buf, 2))
        elif length == 127:
            (length,) = struct.unpack(">Q", _recv_exact(sock, buf, 8))
        mask = _recv_exact(sock, buf, 4) if masked else b""
        payload = _recv_exact(sock, buf, length)
        if masked:
            payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        if opcode in (0x8, 0x9, 0xA):
            return opcode, payload
        if first_opcode is None:
            first_opcode = opcode
        parts.append(payload)
        if fin:
            return first_opcode, b"".join(parts)


class WsRealtimeClient:
    """Pure-stdlib RFC 6455 WebSocket client for /v1/realtime E2E tests."""

    def __init__(
        self,
        host: str,
        port: int,
        path: str = "/v1/realtime",
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.buf = bytearray()
        self._lock = threading.Lock()
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        extra = "".join(f"{k}: {v}\r\n" for k, v in (headers or {}).items())
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"{extra}\r\n"
        ).encode("latin-1")
        self.sock.sendall(req)
        while b"\r\n\r\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                self.sock.close()
                raise ConnectionError("WebSocket handshake closed before headers")
            self.buf.extend(chunk)
        idx = self.buf.index(b"\r\n\r\n") + 4
        head = bytes(self.buf[:idx]).decode("latin-1")
        del self.buf[:idx]
        lines = head.split("\r\n")
        if len(lines[0].split()) < 2 or lines[0].split()[1] != "101":
            self.sock.close()
            raise ConnectionError(f"WebSocket upgrade failed: {lines[0]}")
        hdrs = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:]) if k}
        if hdrs.get("sec-websocket-accept") != _ws_accept_key(key):
            self.sock.close()
            raise ConnectionError("Invalid Sec-WebSocket-Accept header")

    def __enter__(self) -> WsRealtimeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def send_text(self, text: str) -> None:
        with self._lock:
            self.sock.sendall(_ws_encode_frame(0x1, text.encode("utf-8"), masked=True))

    def send_json(self, obj: dict[str, Any]) -> None:
        self.send_text(json.dumps(obj, ensure_ascii=False))

    def send_audio_pcm16(self, pcm_bytes: bytes, *, chunk_bytes: int = 3200) -> None:
        if not pcm_bytes:
            self.send_json({"type": "input_audio_buffer.append", "audio": ""})
            return
        for offset in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[offset : offset + chunk_bytes]
            self.send_json({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            })

    def ping(self, payload: bytes = b"ping") -> bytes:
        with self._lock:
            self.sock.sendall(_ws_encode_frame(0x9, payload, masked=True))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            op, data = _ws_read_frame(self.sock, self.buf)
            if op == 0xA:
                return data
            if op == 0x8:
                raise ConnectionError("WebSocket closed during ping")
        raise TimeoutError("No pong received")

    def recv_json(self, timeout: float = 2.0) -> dict[str, Any]:
        prev = self.sock.gettimeout()
        self.sock.settimeout(timeout)
        try:
            while True:
                op, payload = _ws_read_frame(self.sock, self.buf)
                if op == 0x9:
                    with self._lock:
                        self.sock.sendall(_ws_encode_frame(0xA, payload, masked=True))
                    continue
                if op == 0x8:
                    raise ConnectionError("WebSocket closed by server")
                if op == 0x1:
                    return json.loads(payload.decode("utf-8"))
        finally:
            self.sock.settimeout(prev)

    def close(self) -> None:
        try:
            with self._lock:
                self.sock.sendall(_ws_encode_frame(0x8, struct.pack(">H", 1000), masked=True))
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# FakeUpstreamLLMServer (llama-server simulation)
# ---------------------------------------------------------------------------

class FakeUpstreamLLMServer:
    """In-process fake llama-server serving /health, /v1/chat/completions, and /v1/responses."""

    def __init__(self) -> None:
        self.health_statuses: deque[int] = deque()
        self.chat_replies: deque[str] = deque()
        self.chat_finish: deque[str] = deque()
        self.chat_statuses: deque[int] = deque()
        self.chat_delays: deque[float] = deque()
        self.responses_events: deque[list[dict[str, Any]]] = deque()
        self.responses_json: deque[dict[str, Any]] = deque()
        self.responses_statuses: deque[int] = deque()
        self.responses_delays: deque[float] = deque()
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.port: int = self.server.server_address[1]
        self.base_url: str = f"http://127.0.0.1:{self.port}/v1"
        self.health_url: str = f"http://127.0.0.1:{self.port}/health"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def queue_chat(self, reply: str = "IGNORE", *, finish: str = "stop", status: int = 200, delay_s: float = 0.0) -> None:
        self.chat_replies.append(reply)
        self.chat_finish.append(finish)
        self.chat_statuses.append(status)
        self.chat_delays.append(delay_s)

    def queue_response_text(self, text: str, *, response_id: str | None = None) -> None:
        rid = response_id or f"resp_{uuid.uuid4().hex[:8]}"
        self.responses_events.append([
            {"type": "response.created", "response": {"id": rid, "status": "in_progress", "output": []}},
            {"type": "response.output_text.delta", "delta": text},
            {"type": "response.completed", "response": {"id": rid, "status": "completed", "output": []}},
        ])

    def queue_tool_roundtrip(
        self,
        tool_name: str,
        arguments: dict[str, Any] | str,
        final_text: str,
        *,
        call_id: str | None = None,
    ) -> None:
        cid = call_id or f"call_{uuid.uuid4().hex[:10]}"
        fid = f"fc_{uuid.uuid4().hex[:8]}"
        args_str = arguments if isinstance(arguments, str) else json.dumps(arguments, separators=(",", ":"))
        item = {
            "id": fid,
            "type": "function_call",
            "status": "completed",
            "call_id": cid,
            "name": tool_name,
            "arguments": args_str,
        }
        rid1 = f"resp_{uuid.uuid4().hex[:8]}"
        rid2 = f"resp_{uuid.uuid4().hex[:8]}"
        self.responses_events.append([
            {"type": "response.created", "response": {"id": rid1, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "output_index": 0, "item": {**item, "status": "in_progress", "arguments": ""}},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": fid, "delta": args_str},
            {"type": "response.function_call_arguments.done", "output_index": 0, "item_id": fid, "arguments": args_str},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": {"id": rid1, "status": "completed", "output": [item]}},
        ])
        self.responses_events.append([
            {"type": "response.created", "response": {"id": rid2, "status": "in_progress", "output": []}},
            {"type": "response.output_text.delta", "delta": final_text},
            {"type": "response.completed", "response": {"id": rid2, "status": "completed", "output": []}},
        ])

    def chat_requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self.requests if r["path"].endswith("/chat/completions")]

    def responses_requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self.requests if r["path"].endswith("/responses")]

    def last_instructions(self) -> str:
        reqs = self.responses_requests()
        if not reqs:
            return ""
        return str(reqs[-1]["body"].get("instructions") or "")

    def reset_requests(self) -> None:
        with self._lock:
            self.requests.clear()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def _send_json(self, code: int, payload: dict[str, Any]) -> None:
                out = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def do_GET(self) -> None:
                if self.path.split("?", 1)[0] == "/health":
                    code = outer.health_statuses.popleft() if outer.health_statuses else 200
                    self._send_json(code, {"status": "ok" if code == 200 else "loading"})
                    return
                self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw.decode("utf-8") or "{}")
                hdrs = {k.lower(): v for k, v in self.headers.items()}
                with outer._lock:
                    outer.requests.append({"path": self.path, "headers": hdrs, "body": body})

                if self.path.endswith("/chat/completions"):
                    delay = outer.chat_delays.popleft() if outer.chat_delays else 0.0
                    if delay > 0:
                        time.sleep(delay)
                    status = outer.chat_statuses.popleft() if outer.chat_statuses else 200
                    if status != 200:
                        outer.chat_replies.popleft() if outer.chat_replies else None
                        outer.chat_finish.popleft() if outer.chat_finish else None
                        self._send_json(status, {"error": f"simulated {status}"})
                        return
                    text = outer.chat_replies.popleft() if outer.chat_replies else "IGNORE"
                    finish = outer.chat_finish.popleft() if outer.chat_finish else "stop"
                    self._send_json(200, {
                        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}]
                    })
                    return

                if self.path.endswith("/responses"):
                    delay = outer.responses_delays.popleft() if outer.responses_delays else 0.0
                    if delay > 0:
                        time.sleep(delay)
                    status = outer.responses_statuses.popleft() if outer.responses_statuses else 200
                    if status != 200:
                        self._send_json(status, {"error": f"simulated {status}"})
                        return
                    if not body.get("stream"):
                        payload = outer.responses_json.popleft() if outer.responses_json else {
                            "id": f"resp_{uuid.uuid4().hex[:8]}",
                            "object": "response",
                            "created_at": int(time.time()),
                            "status": "completed",
                            "model": body.get("model", "familiar"),
                            "output": [],
                        }
                        self._send_json(200, payload)
                        return
                    events = outer.responses_events.popleft() if outer.responses_events else [
                        {"type": "response.output_text.delta", "delta": "In character reply."},
                        {"type": "response.completed", "response": {"id": "r1", "status": "completed", "output": []}},
                    ]
                    chunks = b"".join(
                        f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8")
                        for ev in events
                    )
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Content-Length", str(len(chunks)))
                        self.end_headers()
                        self.wfile.write(chunks)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    return

                self._send_json(404, {"error": "not found"})

        return Handler


# ---------------------------------------------------------------------------
# FakeS2SServer (speech-to-speech HTTP + WS /v1/realtime)
# ---------------------------------------------------------------------------

class _ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _parse_http_request(conn: socket.socket) -> tuple[str, str, dict[str, str], bytes, bytearray] | None:
    buf = bytearray()
    conn.settimeout(2.0)
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            return None
        buf.extend(chunk)
        if len(buf) > 1024 * 1024:
            return None
    idx = buf.index(b"\r\n\r\n") + 4
    head = bytes(buf[:idx]).decode("latin-1")
    del buf[:idx]
    lines = head.split("\r\n")
    parts = lines[0].split(" ")
    method = parts[0].upper() if parts else "GET"
    path = parts[1] if len(parts) > 1 else "/"
    headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:]) if k}
    clen = int(headers.get("content-length") or 0)
    while len(buf) < clen:
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf.extend(chunk)
    body = bytes(buf[:clen])
    del buf[:clen]
    return method, path, headers, body, buf


class FakeS2SServer:
    """Pure-stdlib fake speech-to-speech server: HTTP /v1/pool, /v1/usage and WS /v1/realtime."""

    def __init__(self, *, brain_responses_url: str | None = None) -> None:
        self.pool_ready: bool = True
        self.http_requests: list[tuple[str, str]] = []
        self.ws_messages: list[dict[str, Any]] = []
        self.ws_raw_frames: list[str] = []
        self.ws_replies: deque[list[dict[str, Any]]] = deque()
        self.brain_responses_url = brain_responses_url
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ws_conns: list[socket.socket] = []
        self.server = _ReusableTCPServer(("127.0.0.1", 0), self._handler())
        self.port: int = self.server.server_address[1]
        self.base_url: str = f"http://127.0.0.1:{self.port}"
        self.ws_url: str = f"ws://127.0.0.1:{self.port}/v1/realtime"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def broadcast_ws(self, event: dict[str, Any]) -> None:
        frame = _ws_encode_frame(0x1, json.dumps(event, ensure_ascii=False).encode("utf-8"), masked=False)
        with self._lock:
            conns = list(self._ws_conns)
        for conn in conns:
            try:
                conn.sendall(frame)
            except OSError:
                pass

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            for conn in self._ws_conns:
                try:
                    conn.close()
                except OSError:
                    pass
            self._ws_conns.clear()
        self.server.shutdown()
        self.server.server_close()

    def _handler(self):
        outer = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                conn: socket.socket = self.request
                parsed = _parse_http_request(conn)
                if parsed is None:
                    return
                method, raw_path, headers, _body, buf = parsed
                path = raw_path.split("?", 1)[0]

                if headers.get("upgrade", "").lower() == "websocket" and path == "/v1/realtime":
                    key = headers.get("sec-websocket-key", "")
                    accept = _ws_accept_key(key)
                    conn.sendall((
                        "HTTP/1.1 101 Switching Protocols\r\n"
                        "Upgrade: websocket\r\n"
                        "Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                    ).encode("latin-1"))
                    with outer._lock:
                        outer._ws_conns.append(conn)
                    conn.settimeout(0.25)
                    try:
                        while not outer._stop.is_set():
                            try:
                                op, payload = _ws_read_frame(conn, buf)
                            except TimeoutError:
                                continue
                            except (ConnectionError, OSError):
                                break
                            if op == 0x8:
                                try:
                                    conn.sendall(_ws_encode_frame(0x8, struct.pack(">H", 1000)))
                                except OSError:
                                    pass
                                break
                            if op == 0x9:
                                conn.sendall(_ws_encode_frame(0xA, payload))
                                continue
                            if op == 0x1:
                                txt = payload.decode("utf-8", errors="replace")
                                with outer._lock:
                                    outer.ws_raw_frames.append(txt)
                                try:
                                    msg = json.loads(txt)
                                except json.JSONDecodeError:
                                    continue
                                with outer._lock:
                                    outer.ws_messages.append(msg)
                                mtype = msg.get("type")
                                if mtype in ("response.create", "input_audio_buffer.commit"):
                                    if outer.ws_replies:
                                        for ev in outer.ws_replies.popleft():
                                            conn.sendall(_ws_encode_frame(0x1, json.dumps(ev).encode("utf-8")))
                                    elif outer.brain_responses_url and msg.get("transcript"):
                                        resp = requests.post(
                                            outer.brain_responses_url,
                                            json={
                                                "model": "familiar",
                                                "stream": True,
                                                "input": [{"role": "user", "content": msg["transcript"]}],
                                            },
                                            stream=True,
                                            timeout=10.0,
                                        )
                                        for line in resp.iter_lines(decode_unicode=True):
                                            if line and line.startswith("data: "):
                                                ev = json.loads(line[6:])
                                                conn.sendall(_ws_encode_frame(0x1, json.dumps(ev).encode("utf-8")))
                    finally:
                        with outer._lock:
                            if conn in outer._ws_conns:
                                outer._ws_conns.remove(conn)
                    return

                with outer._lock:
                    outer.http_requests.append((method, path))
                if path == "/v1/pool":
                    code = 200 if outer.pool_ready else 503
                    body_out = json.dumps({"pool_size": 1, "ready": outer.pool_ready}).encode("utf-8")
                elif path == "/v1/usage":
                    code = 200
                    with outer._lock:
                        conns = len(outer._ws_conns)
                    body_out = json.dumps({"connections": conns}).encode("utf-8")
                else:
                    code = 404
                    body_out = b'{"error":"not found"}'
                status_text = {200: "OK", 404: "Not Found", 503: "Service Unavailable"}[code]
                conn.sendall((
                    f"HTTP/1.1 {code} {status_text}\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body_out)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("latin-1") + body_out)

        return Handler


# ---------------------------------------------------------------------------
# FakeReachyServer (Reachy Mini daemon HTTP + /rpc JSON-RPC 2.0 WebSocket)
# ---------------------------------------------------------------------------

class FakeReachyServer:
    """Fake Reachy Mini robot daemon (:8000) and conversation app /rpc (:7860)."""

    def __init__(self) -> None:
        self.app_running: bool = True
        self.backend_connected: bool = False
        self.connect_on_config: bool = True
        self.connect_after_polls: int = 0
        self.configured_host: str | None = None
        self.configured_port: int | None = None
        self.applied_personality: str | None = None
        self.applied_voice: str | None = None
        self.enabled_tools: list[str] = []
        self.available_voices: list[str] = ["Aiden", "Serena", "Vivian"]
        self.available_tools: list[str] = [
            "move_head", "play_emotion", "stop_emotion", "idle_do_nothing", "head_tracking", "camera"
        ]
        self.personalities: list[str] = ["default", "familiar"]
        self.daemon_statuses: deque[tuple[int, dict[str, Any]]] = deque()
        self.rpc_errors: dict[str, deque[str]] = {}
        self.daemon_calls: list[tuple[str, str]] = []
        self.rpc_calls: list[dict[str, Any]] = []
        self._poll_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ws_conns: list[socket.socket] = []

        self.daemon_server = ThreadingHTTPServer(("127.0.0.1", 0), self._daemon_handler())
        self.daemon_server.daemon_threads = True
        self.daemon_port: int = self.daemon_server.server_address[1]
        threading.Thread(target=self.daemon_server.serve_forever, daemon=True).start()

        self.rpc_server = _ReusableTCPServer(("127.0.0.1", 0), self._rpc_handler())
        self.rpc_port: int = self.rpc_server.server_address[1]
        threading.Thread(target=self.rpc_server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            for conn in self._ws_conns:
                try:
                    conn.close()
                except OSError:
                    pass
            self._ws_conns.clear()
        self.daemon_server.shutdown()
        self.daemon_server.server_close()
        self.rpc_server.shutdown()
        self.rpc_server.server_close()

    def _daemon_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                out = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                with outer._lock:
                    outer.daemon_calls.append(("GET", path))
                if path == "/api/apps/current-app-status":
                    if outer.daemon_statuses:
                        code, body = outer.daemon_statuses.popleft()
                        self._send(code, body)
                        return
                    info = {"name": "reachy_mini_conversation_app"} if outer.app_running else None
                    self._send(200, {"state": "running" if outer.app_running else "stopped", "info": info})
                    return
                if path == "/api/daemon/status":
                    self._send(200, {"status": "ok"})
                    return
                self._send(404, {"error": "not found"})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                with outer._lock:
                    outer.daemon_calls.append(("POST", path))
                if path.startswith("/api/apps/start-app/"):
                    outer.app_running = True
                    self._send(200, {"ok": True, "app": path.rsplit("/", 1)[-1]})
                    return
                self._send(404, {"error": "not found"})

        return Handler

    def _dispatch_rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any] | list[Any]:
        err_q = self.rpc_errors.get(method)
        if err_q:
            reason = err_q.popleft()
            raise ValueError(reason)
        if method == "conversation.status":
            if self.configured_host is not None and self.connect_on_config:
                self._poll_count += 1
                if self._poll_count >= self.connect_after_polls:
                    self.backend_connected = True
            return {
                "backend_connected": self.backend_connected,
                "host": self.configured_host,
                "port": self.configured_port,
            }
        if method == "backend.config":
            self.configured_host = params.get("hf_host")
            self.configured_port = params.get("hf_port")
            if self.connect_on_config and self.connect_after_polls == 0:
                self.backend_connected = True
            return {"message": f"configured {self.configured_host}:{self.configured_port}"}
        if method == "voices.list":
            return list(self.available_voices)
        if method == "voices.apply":
            self.applied_voice = params.get("voice")
            return {"status": "applied", "voice": self.applied_voice}
        if method == "personalities.list":
            return list(self.personalities)
        if method == "personalities.save":
            name = str(params.get("name", "familiar"))
            if name not in self.personalities:
                self.personalities.append(name)
            return {"value": name}
        if method == "personalities.apply":
            self.applied_personality = params.get("name")
            return {"status": "applied", "startup": bool(params.get("persist"))}
        if method == "profile_tools.get":
            return {
                "profile": self.applied_personality or "default",
                "enabled_tools": list(self.enabled_tools),
                "available_tools": [{"id": t} for t in self.available_tools],
            }
        if method == "profile_tools.save":
            self.enabled_tools = list(params.get("enabled_tools") or [])
            return {"message": "saved", "enabled_tools": self.enabled_tools}
        return {"ok": True}

    def _rpc_handler(self):
        outer = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                conn: socket.socket = self.request
                parsed = _parse_http_request(conn)
                if parsed is None:
                    return
                _method, raw_path, headers, _body, buf = parsed
                path = raw_path.split("?", 1)[0]
                if headers.get("upgrade", "").lower() != "websocket" or path != "/rpc":
                    conn.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                    return
                accept = _ws_accept_key(headers.get("sec-websocket-key", ""))
                conn.sendall((
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                ).encode("latin-1"))
                with outer._lock:
                    outer._ws_conns.append(conn)
                conn.settimeout(0.25)
                try:
                    while not outer._stop.is_set():
                        try:
                            op, payload = _ws_read_frame(conn, buf)
                        except TimeoutError:
                            continue
                        except (ConnectionError, OSError):
                            break
                        if op == 0x8:
                            break
                        if op == 0x9:
                            conn.sendall(_ws_encode_frame(0xA, payload))
                            continue
                        if op != 0x1:
                            continue
                        req = json.loads(payload.decode("utf-8"))
                        method = str(req.get("method", ""))
                        params = req.get("params") or {}
                        rid = req.get("id")
                        with outer._lock:
                            outer.rpc_calls.append({"method": method, "params": params, "id": rid})
                        try:
                            result = outer._dispatch_rpc(method, params)
                            resp = {"jsonrpc": "2.0", "id": rid, "result": result}
                        except ValueError as exc:
                            reason = str(exc)
                            resp = {
                                "jsonrpc": "2.0",
                                "id": rid,
                                "error": {"code": -32000, "message": reason, "data": {"reason": reason}},
                            }
                        conn.sendall(_ws_encode_frame(0x1, json.dumps(resp).encode("utf-8")))
                finally:
                    with outer._lock:
                        if conn in outer._ws_conns:
                            outer._ws_conns.remove(conn)

        return Handler


# ---------------------------------------------------------------------------
# Synthetic multi-speaker PCM16 waveform generators (pure numpy, deterministic)
# ---------------------------------------------------------------------------

def _synthesize_formant_voice(
    *,
    f0_hz: float,
    formants_hz: tuple[float, float, float],
    bandwidths_hz: tuple[float, float, float] = (90.0, 130.0, 180.0),
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    seed: int = 0,
) -> bytes:
    """Generate a deterministic harmonic speech-like PCM16 signal with distinct pitch & formants."""
    n = int(round(duration_s * sample_rate))
    if n <= 0:
        return b""
    t = np.arange(n, dtype=np.float64) / sample_rate
    phase_offset = (seed % 17) * 0.37
    # Gentle natural pitch contour + syllable envelope so delta/delta-delta MFCCs are informative.
    inst_f0 = f0_hz * (1.0 + 0.025 * np.sin(2.0 * math.pi * 2.8 * t + phase_offset))
    phase_acc = 2.0 * math.pi * np.cumsum(inst_f0) / sample_rate
    sig = np.zeros(n, dtype=np.float64)
    nyquist = sample_rate * 0.48
    max_h = max(1, int(nyquist // f0_hz))
    for k in range(1, max_h + 1):
        freq = k * f0_hz
        # Glottal spectral tilt (-12 dB/oct) shaped by 3 Lorentzian formant resonances.
        amp = 1.0 / (k ** 1.1)
        gain = 0.0
        for f_res, bw in zip(formants_hz, bandwidths_hz):
            gain += 1.0 / (1.0 + ((freq - f_res) / (bw * 0.5)) ** 2)
        sig += (amp * gain) * np.sin(k * phase_acc + (k * 0.13 + phase_offset))
    syllable_env = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(2.0 * math.pi * 3.4 * t + phase_offset))
    sig *= syllable_env
    peak = float(np.max(np.abs(sig))) or 1.0
    pcm = np.clip((sig / peak) * 0.65 * 32767.0, -32767.0, 32767.0).astype("<i2")
    return pcm.tobytes()


def brokk_pcm(duration_s: float = 2.0, sample_rate: int = 16000, *, seed: int = 1) -> bytes:
    """Deep baritone voice profile for Brokk."""
    return _synthesize_formant_voice(
        f0_hz=104.0, formants_hz=(430.0, 1220.0, 2280.0), duration_s=duration_s, sample_rate=sample_rate, seed=seed
    )


def till_pcm(duration_s: float = 2.0, sample_rate: int = 16000, *, seed: int = 2) -> bytes:
    """High soprano voice profile for Till."""
    return _synthesize_formant_voice(
        f0_hz=238.0, formants_hz=(790.0, 2180.0, 3180.0), duration_s=duration_s, sample_rate=sample_rate, seed=seed
    )


def dm_pcm(duration_s: float = 2.0, sample_rate: int = 16000, *, seed: int = 3) -> bytes:
    """Mid tenor/baritone voice profile for the DM."""
    return _synthesize_formant_voice(
        f0_hz=152.0, formants_hz=(580.0, 1680.0, 2680.0), duration_s=duration_s, sample_rate=sample_rate, seed=seed
    )


def krag_pcm(duration_s: float = 2.0, sample_rate: int = 16000, *, seed: int = 4) -> bytes:
    """Low gravelly bass voice profile for Krag."""
    return _synthesize_formant_voice(
        f0_hz=86.0, formants_hz=(340.0, 960.0, 1920.0), duration_s=duration_s, sample_rate=sample_rate, seed=seed
    )


def stranger_pcm(duration_s: float = 2.0, sample_rate: int = 16000, *, seed: int = 5) -> bytes:
    """Unrecognized high voice profile outside enrolled centroids."""
    return _synthesize_formant_voice(
        f0_hz=320.0, formants_hz=(940.0, 2550.0, 3520.0), duration_s=duration_s, sample_rate=sample_rate, seed=seed
    )


def silence_pcm(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Pure digital silence PCM16."""
    n = int(round(duration_s * sample_rate))
    return b"\x00\x00" * max(0, n)


def pcm16_to_b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def pcm16_to_float_samples(pcm: bytes) -> list[float]:
    if not pcm:
        return []
    arr = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return arr.tolist()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
