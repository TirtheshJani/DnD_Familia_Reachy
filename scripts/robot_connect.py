"""Point the robot's conversation app at this laptop and select the familiar profile.

The app running on the robot serves only / and /static on port 7860. Its settings UI
talks JSON-RPC over a WebSocket at ws://<robot>:7860/rpc, so this script does the same.
Method names and payloads come from the UI source it serves (checked 2026-09-16):
static/js/api.js (rpcCall and the method table), views/settings.js (backend.config),
views/home.js and components/profile-modal.js (personalities.save and .apply),
views/tools.js with components/profile-tools.js (profile_tools.get and .save).
That api.js is byte-identical to the Hugging Face Space main branch (pyproject 1.0.1),
although the daemon's app listing still records Space sha 8d9fe4cc, which is 0.10.0
and used REST routes under /api/v1. Nothing here needs SSH:

  1. start the app through the daemon if it is not already running, wait for /rpc
  2. backend.config         local mode, <laptop LAN IP>:8765
  3. personalities.save     the familiar profile, then profile_tools.save with exactly
                            robot_profile/familiar/tools.txt, then personalities.apply
                            with persist so it is selected at the next start too
  4. voices.apply           Aiden, if voices.list offers it
  5. print conversation.status

Run after scripts/start_all.sh prints "Point the robot at", from Windows or WSL:

    uv run python scripts/robot_connect.py            # host IP detected
    uv run python scripts/robot_connect.py --host-ip 10.0.0.95
    uv run python scripts/robot_connect.py --dry-run  # read-only, changes nothing

The laptop's address comes from DHCP and moves (docs/versions.md), so run this again
whenever it changes. The robot can only reach port 8765 once the Hyper-V firewall rule
from docs/versions.md exists.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

import requests

APP = "reachy_mini_conversation_app"
PROFILE = "familiar"
PROFILE_DIR = Path(__file__).resolve().parent.parent / "robot_profile" / PROFILE
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class RpcError(Exception):
    def __init__(self, method: str, error: dict) -> None:
        self.reason = (error.get("data") or {}).get("reason")
        super().__init__(f"{method} failed: {error.get('message')} (reason {self.reason})")


class RpcSocket:
    """JSON-RPC over a WebSocket with nothing but the standard library.

    The project has no websocket dependency and this script makes a dozen calls, so it
    speaks RFC 6455 directly: one HTTP Upgrade handshake, masked text frames out, server
    frames of any length in, pings answered with pongs. Replies are matched by id;
    notifications the app broadcasts meanwhile ({method, params}, no id) are skipped.
    """

    def __init__(self, host: str, port: int, path: str = "/rpc", timeout: float = 10) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.buf = b""
        self.ids = 0
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        ).encode())
        while b"\r\n\r\n" not in self.buf:
            self._fill()
        head, self.buf = self.buf.split(b"\r\n\r\n", 1)
        lines = head.decode("latin-1").split("\r\n")
        headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        if lines[0].split()[1:2] != ["101"] or headers.get("sec-websocket-accept") != accept:
            self.sock.close()
            raise ConnectionError(f"ws://{host}:{port}{path} refused the upgrade: {lines[0]}")

    def __enter__(self) -> RpcSocket:
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self._send(0x8, struct.pack(">H", 1000))
        except OSError:
            pass
        self.sock.close()

    def _fill(self) -> None:
        chunk = self.sock.recv(65536)
        if not chunk:
            raise ConnectionError("the robot closed the /rpc socket")
        self.buf += chunk

    def _take(self, n: int) -> bytes:
        while len(self.buf) < n:
            self._fill()
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _send(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        if n < 126:
            head = struct.pack(">BB", 0x80 | opcode, 0x80 | n)
        elif n < 65536:
            head = struct.pack(">BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            head = struct.pack(">BBQ", 0x80 | opcode, 0x80 | 127, n)
        mask = os.urandom(4)
        self.sock.sendall(head + mask + bytes(b ^ mask[i & 3] for i, b in enumerate(payload)))

    def _message(self) -> str:
        """The next complete text message, answering pings and joining fragments."""
        parts: list[bytes] = []
        while True:
            b0, b1 = self._take(2)
            n = b1 & 0x7F
            if n == 126:
                (n,) = struct.unpack(">H", self._take(2))
            elif n == 127:
                (n,) = struct.unpack(">Q", self._take(8))
            mask = self._take(4) if b1 & 0x80 else b""
            payload = self._take(n)
            if mask:
                payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
            opcode = b0 & 0x0F
            if opcode == 0x8:
                raise ConnectionError("the robot closed the /rpc socket")
            if opcode == 0x9:
                self._send(0xA, payload)
            elif opcode in (0x0, 0x1, 0x2):
                parts.append(payload)
                if b0 & 0x80:
                    return b"".join(parts).decode("utf-8")

    def call(self, method: str, params: dict | None = None, timeout: float = 30) -> Any:
        self.ids += 1
        rid = f"robot_connect-{self.ids}"
        body = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        self._send(0x1, json.dumps(body).encode("utf-8"))
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"{method} got no reply within {timeout:.0f} s")
            self.sock.settimeout(left)
            msg = json.loads(self._message())
            if msg.get("id") != rid:
                continue
            if msg.get("error"):
                raise RpcError(method, msg["error"])
            return msg.get("result")


def detect_host_ip(robot: str) -> str:
    """The local address the OS would use to reach the robot. No packet is sent."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((robot, 8000))
        return s.getsockname()[0]


def read_tools(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def connect_when_ready(robot: str, seconds: float) -> tuple[RpcSocket, dict]:
    """Open /rpc once it answers conversation.status."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            rpc = RpcSocket(robot, 7860)
            try:
                return rpc, rpc.call("conversation.status", timeout=10)
            except Exception:
                rpc.sock.close()
                raise
        except (OSError, RpcError, ValueError) as exc:
            if time.monotonic() > deadline:
                sys.exit(f"ws://{robot}:7860/rpc did not answer within {seconds:.0f} s: {exc}")
            time.sleep(2)


def call_when_loop_up(rpc: RpcSocket, method: str, params: dict, seconds: float = 60) -> Any:
    """Retry while the app answers loop_unavailable (its event loop is still starting)."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            return rpc.call(method, params)
        except RpcError as exc:
            if exc.reason != "loop_unavailable" or time.monotonic() > deadline:
                raise
            time.sleep(2)


def show(label: str, value: object) -> None:
    print(f"{label}:\n{json.dumps(value, indent=2)}")


def dry_run(rpc: RpcSocket, status: dict) -> None:
    show("conversation.status", status)
    show("personalities.list", rpc.call("personalities.list"))
    show("voices.list", rpc.call("voices.list"))
    tools = rpc.call("profile_tools.get")
    tools["available_tools"] = [t.get("id") for t in tools.get("available_tools", [])]
    show("profile_tools.get (current profile, available_tools shown as ids)", tools)


def configure(rpc: RpcSocket, host_ip: str, port: int, wanted_voice: str) -> None:
    result = rpc.call("backend.config", {"hf_mode": "local", "hf_host": host_ip, "hf_port": port})
    print(f"backend.config local {host_ip}:{port}: {result.get('message')}")

    voices = rpc.call("voices.list") or []
    voice = next((v for v in voices if v.lower() == wanted_voice.lower()), None)
    if voice is None:
        print(f"voice {wanted_voice!r} is not offered ({', '.join(voices)}); keeping the app default")

    tools = read_tools(PROFILE_DIR / "tools.txt")
    saved = rpc.call("personalities.save", {
        "name": PROFILE,
        "instructions": (PROFILE_DIR / "instructions.md").read_text(encoding="utf-8"),
        "voice": voice or "",
        "tools_text": "\n".join(tools),
        "overwrite": True,
    })
    profile = saved["value"]
    print(f"personalities.save: {profile}")
    result = rpc.call("profile_tools.save", {"profile": profile, "enabled_tools": tools})
    print(f"profile_tools.save: {result.get('message')} enabled {result.get('enabled_tools')}")
    result = call_when_loop_up(rpc, "personalities.apply", {"name": profile, "persist": True, "force": True})
    print(f"personalities.apply: {result.get('status')} (startup: {result.get('startup')})")

    if voice:
        deadline = time.monotonic() + 30
        while not rpc.call("conversation.status").get("backend_connected") and time.monotonic() < deadline:
            time.sleep(2)
        result = call_when_loop_up(rpc, "voices.apply", {"voice": voice})
        print(f"voices.apply {voice}: {result.get('status')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", default="10.0.0.100")
    ap.add_argument("--host-ip", default=None, help="laptop LAN address the robot connects to")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--voice", default="Aiden")
    ap.add_argument("--dry-run", action="store_true",
                    help="connect and print the app's current state without changing anything")
    args = ap.parse_args()

    daemon = f"http://{args.robot}:8000/api/apps"
    current = requests.get(f"{daemon}/current-app-status", timeout=5).json() or {}
    running = (current.get("info") or {}).get("name")
    print(f"daemon: current app {running or 'none'} ({current.get('state')})")
    if running != APP:
        if args.dry_run:
            sys.exit(f"{APP} is not running; --dry-run does not start it")
        print(f"starting {APP} on {args.robot}")
        r = requests.post(f"{daemon}/start-app/{APP}", timeout=30)
        if not r.ok:
            sys.exit(f"start-app failed ({r.status_code}): {r.text}")

    rpc, status = connect_when_ready(args.robot, 180)
    with rpc:
        if args.dry_run:
            print(f"would point the backend at {args.host_ip or detect_host_ip(args.robot)}:{args.port}")
            dry_run(rpc, status)
            return
        try:
            configure(rpc, args.host_ip or detect_host_ip(args.robot), args.port, args.voice)
            show("conversation.status", rpc.call("conversation.status"))
        except (RpcError, TimeoutError, ConnectionError) as exc:
            sys.exit(str(exc))


if __name__ == "__main__":
    main()
