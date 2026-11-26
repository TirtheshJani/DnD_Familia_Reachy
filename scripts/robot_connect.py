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
  4. voices.apply           bm_fable (or configured voice), if voices.list offers it
  5. verify conversation.status reports backend_connected == True

Run after scripts/start_all.sh prints "Point the robot at", from Windows or WSL:

    uv run python scripts/robot_connect.py            # auto-discovers reachy-mini.local -> 10.0.0.100
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
import ipaddress
import json
import logging
import os
import re
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import requests

APP = "reachy_mini_conversation_app"
PROFILE = "familiar"
PROFILE_DIR = Path(__file__).resolve().parent.parent / "robot_profile" / PROFILE
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
DEFAULT_ROBOT_CANDIDATES = ("reachy-mini.local", "10.0.0.100")
TRANSIENT_DAEMON_STATES = {"starting", "initializing", "stopping", "restarting", "pending"}
# First Pollen release that drops a dead realtime socket when a mic send stalls (5 s timeout);
# 1.0.1 and 1.1.x can freeze their record loop for minutes. Release notes, v1.2.0.
MIN_APP_VERSION = (1, 2, 0)

log = logging.getLogger("robot_connect")


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


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _resolve_http_fns(
    session: Any = None,
    http_get: Callable[..., Any] | None = None,
    http_post: Callable[..., Any] | None = None,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    get_fn = http_get or (session.get if session is not None else requests.get)
    post_fn = http_post or (session.post if session is not None else requests.post)
    return get_fn, post_fn


def resolve_robot_host(
    candidates: Sequence[str] = DEFAULT_ROBOT_CANDIDATES,
    *,
    preferred_host: str | None = None,
    fallback_host: str | None = "10.0.0.100",
    allow_fallback: bool = False,
    resolver: Callable[..., Any] = socket.getaddrinfo,
    http_get: Callable[..., Any] | None = None,
    session: Any = None,
    daemon_port: int = 8000,
    port: int | None = None,
    timeout: float = 3.0,
) -> str:
    """Resolve reachy-mini.local first and fall back to IP if DNS/HTTP probe fails."""
    effective_port = port if port is not None else daemon_port
    get_fn, _ = _resolve_http_fns(session=session, http_get=http_get)

    ordered: list[str] = []
    preferred = preferred_host.strip() if preferred_host is not None else ""
    explicit_preferred = bool(preferred and preferred != "auto")
    if explicit_preferred:
        ordered.append(preferred)
        if allow_fallback and fallback_host and fallback_host not in ordered:
            ordered.append(fallback_host)
    else:
        ordered.extend(candidates)
        if fallback_host and fallback_host not in ordered:
            ordered.append(fallback_host)

    first_resolvable: str | None = None
    for candidate in ordered:
        resolvable = _is_ip_literal(candidate)
        if not resolvable:
            try:
                infos = resolver(candidate, effective_port)
                resolvable = bool(infos)
            except OSError:
                resolvable = False
        if not resolvable:
            continue
        if first_resolvable is None:
            first_resolvable = candidate
        try:
            resp = get_fn(f"http://{candidate}:{effective_port}/api/apps/current-app-status", timeout=timeout)
            status_code = getattr(resp, "status_code", 200)
            if status_code < 500:
                return candidate
        except (requests.RequestException, OSError, ValueError):
            continue

    if first_resolvable is not None:
        return first_resolvable
    raise ConnectionError(f"Could not resolve any robot host from {ordered}")


def parse_app_version(listing: dict) -> tuple[int, int, int] | None:
    """The app version in a daemon app listing or status dict, or None if absent or unreadable."""
    if not isinstance(listing, dict):
        return None
    info = listing.get("info")
    raw = listing.get("version") or (info.get("version") if isinstance(info, dict) else None)
    match = re.match(r"\s*v?(\d+)\.(\d+)\.(\d+)", str(raw or ""))
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def _check_app_version(*sources: dict) -> None:
    """Warn, never raise, when the robot app is older than MIN_APP_VERSION or its version is unknown."""
    version = next((v for v in map(parse_app_version, sources) if v), None)
    want = ".".join(map(str, MIN_APP_VERSION))
    if version is None:
        log.warning("could not read the %s version; it needs %s or newer for the 5 s mic send timeout", APP, want)
    elif version < MIN_APP_VERSION:
        log.warning("%s %s is older than %s: it has no mic send timeout, so a dead network path can freeze "
                    "its mic for minutes. Upgrade the app on the robot.", APP, ".".join(map(str, version)), want)


def detect_host_ip(robot: str, port: int = 8000) -> str:
    """The local address the OS would use to reach the robot. No packet is sent."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((robot, port))
        return s.getsockname()[0]


def read_tools(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def connect_when_ready(
    robot: str,
    seconds: float,
    *,
    rpc_port: int = 7860,
    rpc_factory: Callable[..., Any] = RpcSocket,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, dict]:
    """Open /rpc once it answers conversation.status."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            rpc = rpc_factory(robot, rpc_port)
            try:
                return rpc, rpc.call("conversation.status", timeout=10) or {}
            except Exception:
                sock = getattr(rpc, "sock", None)
                if sock is not None:
                    sock.close()
                elif hasattr(rpc, "close"):
                    rpc.close()
                raise
        except (OSError, RpcError, ValueError, TimeoutError, ConnectionError) as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"ws://{robot}:{rpc_port}/rpc did not answer within {seconds:.0f} s: {exc}") from exc
            sleep(2)


def call_when_loop_up(
    rpc: Any,
    method: str,
    params: dict,
    seconds: float = 60,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Retry while the app answers loop_unavailable (its event loop is still starting)."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            return rpc.call(method, params)
        except RpcError as exc:
            if exc.reason != "loop_unavailable" or time.monotonic() >= deadline:
                raise
            sleep(2)


def show(label: str, value: object) -> None:
    print(f"{label}:\n{json.dumps(value, indent=2)}")


def dry_run(rpc: Any, status: dict) -> None:
    show("conversation.status", status)
    show("personalities.list", rpc.call("personalities.list"))
    show("voices.list", rpc.call("voices.list"))
    tools = rpc.call("profile_tools.get") or {}
    tools["available_tools"] = [t.get("id") for t in tools.get("available_tools", [])]
    show("profile_tools.get (current profile, available_tools shown as ids)", tools)


def _poll_daemon_status(
    daemon_url: str,
    *,
    get_fn: Callable[..., Any],
    timeout_s: float,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    deadline = time.monotonic() + max(timeout_s, 1.0)
    last_exc: Exception | None = None
    while True:
        try:
            resp = get_fn(f"{daemon_url}/current-app-status", timeout=5)
            status_code = getattr(resp, "status_code", 200)
            if status_code >= 500:
                raise requests.HTTPError(f"daemon status returned {status_code}")
            current = resp.json() or {}
            state = str(current.get("state") or "").lower()
            if state in TRANSIENT_DAEMON_STATES and time.monotonic() < deadline:
                sleep(2)
                continue
            return current
        except (requests.RequestException, OSError, ValueError) as exc:
            last_exc = exc
            if time.monotonic() >= deadline:
                raise ConnectionError(f"daemon status failed at {daemon_url}: {last_exc}") from exc
            sleep(2)


def _start_app_with_retry(
    daemon_url: str,
    robot: str,
    *,
    post_fn: Callable[..., Any],
    timeout_s: float,
    sleep: Callable[[float], None],
) -> None:
    deadline = time.monotonic() + max(timeout_s, 1.0)
    while True:
        try:
            resp = post_fn(f"{daemon_url}/start-app/{APP}", timeout=30)
            status_code = getattr(resp, "status_code", 200)
            if getattr(resp, "ok", status_code < 400):
                return
            if status_code in (409, 425, 500, 502, 503, 504) and time.monotonic() < deadline:
                sleep(2)
                continue
            text = getattr(resp, "text", "")
            raise ConnectionError(f"start-app failed ({status_code}) on {robot}: {text}")
        except (requests.RequestException, OSError) as exc:
            if time.monotonic() >= deadline:
                raise ConnectionError(f"start-app request failed on {robot}: {exc}") from exc
            sleep(2)


def configure(
    rpc: Any,
    host_ip: str,
    port: int,
    wanted_voice: str,
    *,
    verify_timeout_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    current_status: dict | None = None,
) -> tuple[dict[str, Any], str | None]:
    current = current_status or {}
    skipped = bool(current.get("backend_connected") and current.get("host") == host_ip and current.get("port") == port)
    if skipped:
        # backend.config restarts the backend; skip it when it already points here and is connected.
        print(f"backend.config: already connected to {host_ip}:{port}, not reconfiguring")
    else:
        result = rpc.call("backend.config", {"hf_mode": "local", "hf_host": host_ip, "hf_port": port}) or {}
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
    result = rpc.call("profile_tools.save", {"profile": profile, "enabled_tools": tools}) or {}
    print(f"profile_tools.save: {result.get('message')} enabled {result.get('enabled_tools')}")
    result = call_when_loop_up(
        rpc,
        "personalities.apply",
        # force restarts the backend once when backend.config was skipped, so Reconnect still
        # unsticks an app that reports connected; never twice.
        {"name": profile, "persist": True, **({"force": True} if skipped else {})},
        sleep=sleep,
    ) or {}
    print(f"personalities.apply: {result.get('status')} (startup: {result.get('startup')})")

    # Unconditionally verify backend_connected before declaring success (and before voices.apply).
    deadline = time.monotonic() + max(verify_timeout_s, 0.0)
    status = rpc.call("conversation.status") or {}
    while not status.get("backend_connected") and time.monotonic() < deadline:
        sleep(2)
        status = rpc.call("conversation.status") or {}

    if voice and status.get("backend_connected"):
        v_res = call_when_loop_up(rpc, "voices.apply", {"voice": voice}, sleep=sleep) or {}
        print(f"voices.apply {voice}: {v_res.get('status')}")
        status = rpc.call("conversation.status") or status

    return status, voice


def check_robot_status(
    host: str | None = None,
    *,
    http_get: Callable[..., Any] | None = None,
    session: Any = None,
    rpc_factory: Callable[..., Any] = RpcSocket,
    resolver: Callable[..., Any] = socket.getaddrinfo,
    daemon_port: int = 8000,
    rpc_port: int = 7860,
    fallback_host: str | None = "10.0.0.100",
    allow_fallback: bool = False,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Read-only check of daemon app status and conversation backend connectivity."""
    get_fn, _ = _resolve_http_fns(session=session, http_get=http_get)
    try:
        resolved_host = resolve_robot_host(
            preferred_host=host,
            fallback_host=fallback_host,
            allow_fallback=allow_fallback,
            resolver=resolver,
            http_get=get_fn,
            daemon_port=daemon_port,
            timeout=min(timeout, 3.0),
        )
    except ConnectionError as exc:
        return {
            "ok": False,
            "host": host or (fallback_host if allow_fallback or not host or host == "auto" else "") or "",
            "daemon_port": daemon_port,
            "rpc_port": rpc_port,
            "app_running": False,
            "app_state": None,
            "backend_connected": False,
            "status": None,
            "detail": str(exc),
            "error": str(exc),
        }

    daemon_url = f"http://{resolved_host}:{daemon_port}/api/apps"
    try:
        resp = get_fn(f"{daemon_url}/current-app-status", timeout=timeout)
        status_code = getattr(resp, "status_code", 200)
        if status_code >= 400:
            raise requests.HTTPError(f"daemon returned HTTP {status_code}")
        current = resp.json() or {}
    except (requests.RequestException, OSError, ValueError) as exc:
        return {
            "ok": False,
            "host": resolved_host,
            "daemon_port": daemon_port,
            "rpc_port": rpc_port,
            "app_running": False,
            "app_state": None,
            "backend_connected": False,
            "status": None,
            "detail": f"daemon unreachable at {resolved_host}:{daemon_port}: {exc}",
            "error": str(exc),
        }

    app_name = (current.get("info") or {}).get("name")
    app_state = current.get("state")
    app_running = app_name == APP and str(app_state or "running").lower() not in {"stopped", "error"}
    if not app_running:
        return {
            "ok": False,
            "host": resolved_host,
            "daemon_port": daemon_port,
            "rpc_port": rpc_port,
            "app_running": False,
            "app_state": app_state,
            "backend_connected": False,
            "status": None,
            "detail": f"app={app_name or 'none'} state={app_state}",
            "error": f"{APP} is not running (current={app_name or 'none'}, state={app_state})",
        }

    try:
        with rpc_factory(resolved_host, rpc_port, timeout=timeout) as rpc:
            status = rpc.call("conversation.status", timeout=timeout) or {}
    except (OSError, RpcError, ValueError, TimeoutError, ConnectionError) as exc:
        return {
            "ok": False,
            "host": resolved_host,
            "daemon_port": daemon_port,
            "rpc_port": rpc_port,
            "app_running": True,
            "app_state": app_state,
            "backend_connected": False,
            "status": None,
            "detail": f"rpc error at {resolved_host}:{rpc_port}: {exc}",
            "error": str(exc),
        }

    backend_connected = bool(status.get("backend_connected"))
    return {
        "ok": backend_connected,
        "host": resolved_host,
        "daemon_port": daemon_port,
        "rpc_port": rpc_port,
        "app_running": True,
        "app_state": app_state,
        "backend_connected": backend_connected,
        "status": status,
        "detail": f"app={app_name} state={app_state} backend_connected={backend_connected}",
        "error": None if backend_connected else "backend_connected is False",
    }


def connect_and_configure(
    host: str | None = None,
    *,
    host_ip: str | None = None,
    port: int = 8765,
    voice: str = "bm_fable",
    daemon_port: int = 8000,
    rpc_port: int = 7860,
    fallback_host: str | None = "10.0.0.100",
    allow_fallback: bool = False,
    timeout_s: float = 180.0,
    verify_timeout_s: float = 30.0,
    dry_run_mode: bool = False,
    http_get: Callable[..., Any] | None = None,
    http_post: Callable[..., Any] | None = None,
    session: Any = None,
    rpc_factory: Callable[..., Any] = RpcSocket,
    resolver: Callable[..., Any] = socket.getaddrinfo,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Discover robot, ensure app is running, configure backend/profile, and verify backend_connected."""
    get_fn, post_fn = _resolve_http_fns(session=session, http_get=http_get, http_post=http_post)
    try:
        resolved_host = resolve_robot_host(
            preferred_host=host,
            fallback_host=fallback_host,
            allow_fallback=allow_fallback,
            resolver=resolver,
            http_get=get_fn,
            daemon_port=daemon_port,
            timeout=3.0,
        )
    except ConnectionError as exc:
        return {
            "ok": False,
            "backend_connected": False,
            "host": host or (fallback_host if allow_fallback or not host or host == "auto" else "") or "",
            "host_ip": host_ip,
            "port": port,
            "daemon_port": daemon_port,
            "rpc_port": rpc_port,
            "voice": None,
            "status": None,
            "error": str(exc),
        }

    daemon = f"http://{resolved_host}:{daemon_port}/api/apps"
    try:
        current = _poll_daemon_status(
            daemon,
            get_fn=get_fn,
            timeout_s=min(timeout_s, 30.0),
            sleep=sleep,
        )
    except ConnectionError as exc:
        return {
            "ok": False,
            "backend_connected": False,
            "host": resolved_host,
            "host_ip": host_ip,
            "port": port,
            "daemon_port": daemon_port,
            "rpc_port": rpc_port,
            "voice": None,
            "status": None,
            "error": str(exc),
        }

    running = (current.get("info") or {}).get("name")
    state = str(current.get("state") or "").lower()
    print(f"daemon: current app {running or 'none'} ({current.get('state')})")
    if running != APP or state in {"stopped", "error"}:
        if dry_run_mode:
            return {
                "ok": False,
                "backend_connected": False,
                "host": resolved_host,
                "host_ip": host_ip,
                "port": port,
                "daemon_port": daemon_port,
                "rpc_port": rpc_port,
                "voice": None,
                "status": None,
                "error": f"{APP} is not running; --dry-run does not start it",
            }
        print(f"starting {APP} on {resolved_host}")
        try:
            _start_app_with_retry(
                daemon,
                resolved_host,
                post_fn=post_fn,
                timeout_s=min(timeout_s, 60.0),
                sleep=sleep,
            )
        except ConnectionError as exc:
            return {
                "ok": False,
                "backend_connected": False,
                "host": resolved_host,
                "host_ip": host_ip,
                "port": port,
                "daemon_port": daemon_port,
                "rpc_port": rpc_port,
                "voice": None,
                "status": None,
                "error": str(exc),
            }

    try:
        rpc, status = connect_when_ready(
            resolved_host,
            timeout_s,
            rpc_port=rpc_port,
            rpc_factory=rpc_factory,
            sleep=sleep,
        )
    except (TimeoutError, OSError, ConnectionError, RpcError) as exc:
        return {
            "ok": False,
            "backend_connected": False,
            "host": resolved_host,
            "host_ip": host_ip,
            "port": port,
            "daemon_port": daemon_port,
            "rpc_port": rpc_port,
            "voice": None,
            "status": None,
            "error": str(exc),
        }

    # The daemon status describes whichever app was running before start-app; only trust it for ours.
    _check_app_version(status or {}, current if ((current or {}).get("info") or {}).get("name") == APP else {})
    with rpc:
        effective_ip = host_ip or detect_host_ip(resolved_host, port=daemon_port)
        if dry_run_mode:
            print(f"would point the backend at {effective_ip}:{port}")
            dry_run(rpc, status)
            connected = bool((status or {}).get("backend_connected"))
            return {
                "ok": True,
                "dry_run": True,
                "backend_connected": connected,
                "host": resolved_host,
                "host_ip": effective_ip,
                "port": port,
                "daemon_port": daemon_port,
                "rpc_port": rpc_port,
                "voice": None,
                "status": status,
                "error": None,
            }
        try:
            final_status, applied_voice = configure(
                rpc,
                effective_ip,
                port,
                voice,
                verify_timeout_s=verify_timeout_s,
                sleep=sleep,
                current_status=status,
            )
            show("conversation.status", final_status)
        except (RpcError, TimeoutError, ConnectionError, OSError) as exc:
            return {
                "ok": False,
                "backend_connected": False,
                "host": resolved_host,
                "host_ip": effective_ip,
                "port": port,
                "daemon_port": daemon_port,
                "rpc_port": rpc_port,
                "voice": None,
                "status": None,
                "error": str(exc),
            }

    connected = bool((final_status or {}).get("backend_connected"))
    return {
        "ok": connected,
        "backend_connected": connected,
        "host": resolved_host,
        "host_ip": effective_ip,
        "port": port,
        "daemon_port": daemon_port,
        "rpc_port": rpc_port,
        "voice": applied_voice,
        "status": final_status,
        "error": None if connected else "backend_connected remained False after configuration",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", default="auto", help="robot hostname/IP, or 'auto' for reachy-mini.local -> fallback")
    ap.add_argument("--fallback-host", default=None,
                    help="fallback robot IP if primary/mDNS fails (defaults to 10.0.0.100 in --robot auto mode)")
    ap.add_argument("--daemon-port", type=int, default=8000, help="Reachy Mini daemon HTTP port")
    ap.add_argument("--rpc-port", type=int, default=7860, help="conversation app WebSocket RPC port")
    ap.add_argument("--host-ip", default=None, help="laptop LAN address the robot connects to")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--voice", default="bm_fable")
    ap.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for daemon/RPC readiness")
    ap.add_argument("--verify-timeout", "--connect-timeout", dest="verify_timeout", type=float, default=30.0,
                    help="seconds to wait for backend_connected=True")
    ap.add_argument("--dry-run", action="store_true",
                    help="connect and print the app's current state without changing anything")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    robot_arg = args.robot.strip()
    if not robot_arg:
        ap.error("--robot must not be empty; use 'auto' or a hostname/IP")

    preferred_host = None if robot_arg == "auto" else robot_arg
    if preferred_host is None:
        fallback_host = args.fallback_host if args.fallback_host is not None else "10.0.0.100"
        allow_fallback = False
    else:
        allow_fallback = bool(args.fallback_host)
        fallback_host = args.fallback_host or None

    result = connect_and_configure(
        preferred_host,
        host_ip=args.host_ip,
        port=args.port,
        voice=args.voice,
        daemon_port=args.daemon_port,
        rpc_port=args.rpc_port,
        fallback_host=fallback_host or None,
        allow_fallback=allow_fallback,
        timeout_s=args.timeout,
        verify_timeout_s=args.verify_timeout,
        dry_run_mode=args.dry_run,
    )
    if args.dry_run:
        if not result.get("ok"):
            sys.exit(result.get("error") or "dry-run failed")
        return 0
    if not result.get("backend_connected"):
        sys.exit(result.get("error") or "backend_connected remained False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
