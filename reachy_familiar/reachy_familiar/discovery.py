"""LAN discovery and Wake Daemon client for finding the Zephyrus RTX 4080 laptop."""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("reachy_familiar.discovery")

DISCOVER_MAGIC = b"REACHY_FAMILIAR_DISCOVER"
DEFAULT_WAKE_PORT = 8091
DEFAULT_DISCOVER_PORT = 8092
DEFAULT_BRAIN_PORT = 8090
DEFAULT_WS_PORT = 8765
CACHE_PATH = Path.home() / ".reachy_familiar_host.json"


@dataclass(frozen=True)
class LaptopEndpoint:
    host: str
    wake_port: int = DEFAULT_WAKE_PORT
    brain_port: int = DEFAULT_BRAIN_PORT
    ws_port: int = DEFAULT_WS_PORT

    @property
    def wake_url(self) -> str:
        return f"http://{self.host}:{self.wake_port}"

    @property
    def brain_url(self) -> str:
        return f"http://{self.host}:{self.brain_port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.ws_port}/v1/realtime"


def load_cached_host(path: Path = CACHE_PATH) -> LaptopEndpoint | None:
    try:
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        host = str(raw.get("host") or "").strip()
        if not host:
            return None
        return LaptopEndpoint(
            host=host,
            wake_port=int(raw.get("wake_port") or DEFAULT_WAKE_PORT),
            brain_port=int(raw.get("brain_port") or DEFAULT_BRAIN_PORT),
            ws_port=int(raw.get("ws_port") or DEFAULT_WS_PORT),
        )
    except Exception:
        return None


def save_cached_host(ep: LaptopEndpoint, path: Path = CACHE_PATH) -> None:
    try:
        path.write_text(
            json.dumps(
                {
                    "host": ep.host,
                    "wake_port": ep.wake_port,
                    "brain_port": ep.brain_port,
                    "ws_port": ep.ws_port,
                    "saved_at": time.time(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def _active_broadcast_targets() -> list[str]:
    """Compute broadcast addresses for all active IPv4 interfaces (works on any Wi-Fi or phone hotspot)."""
    targets: list[str] = ["255.255.255.255"]
    try:
        out = subprocess.check_output(["ip", "-4", "-o", "addr", "show"], text=True, timeout=1.5)
        for line in out.splitlines():
            parts = line.split()
            for tok in parts:
                if "/" in tok:
                    try:
                        iface = ipaddress.IPv4Interface(tok)
                        if not iface.ip.is_loopback and not iface.ip.is_link_local:
                            bcast = str(iface.network.broadcast_address)
                            if bcast not in targets:
                                targets.append(bcast)
                    except ValueError:
                        continue
    except Exception:
        pass
    for fallback in ("10.0.0.255", "192.168.1.255", "192.168.0.255", "172.20.10.15", "192.168.43.255"):
        if fallback not in targets:
            targets.append(fallback)
    return targets


def discover_laptop(
    *,
    discover_port: int = DEFAULT_DISCOVER_PORT,
    timeout_s: float = 3.0,
    explicit_host: str | None = None,
    cache_path: Path = CACHE_PATH,
) -> LaptopEndpoint | None:
    """Find the laptop Wake Daemon via explicit env/arg, UDP broadcast, or cached IP."""
    env_host = (explicit_host or os.environ.get("REACHY_FAMILIAR_HOST") or "").strip()
    if env_host:
        ep = LaptopEndpoint(host=env_host)
        if _probe_wake_or_brain(ep):
            save_cached_host(ep, cache_path)
            return ep

    # UDP LAN broadcast discovery (immune to DHCP IP moves).
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(min(1.0, timeout_s))
        try:
            targets = _active_broadcast_targets()
            cached = load_cached_host(cache_path)
            if cached and cached.host not in targets:
                targets.insert(0, cached.host)
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                for tgt in targets:
                    try:
                        sock.sendto(DISCOVER_MAGIC, (tgt, discover_port))
                    except OSError:
                        pass
                try:
                    data, addr = sock.recvfrom(2048)
                    info = json.loads(data.decode("utf-8"))
                    if info.get("service") == "reachy_familiar_wake":
                        host = str(info.get("host") or addr[0]).strip()
                        ep = LaptopEndpoint(
                            host=host,
                            wake_port=int(info.get("wake_port") or DEFAULT_WAKE_PORT),
                            brain_port=int(info.get("brain_port") or DEFAULT_BRAIN_PORT),
                            ws_port=int(info.get("ws_port") or DEFAULT_WS_PORT),
                        )
                        save_cached_host(ep, cache_path)
                        return ep
                except socket.timeout:
                    continue
                except Exception:
                    continue
        finally:
            sock.close()
    except OSError as exc:
        log.debug("UDP discovery failed: %s", exc)

    # Fallback: probe cached host directly (works even if UDP broadcast is filtered).
    cached = load_cached_host(cache_path)
    if cached and _probe_wake_or_brain(cached):
        return cached
    if env_host:
        return LaptopEndpoint(host=env_host)
    return cached


def _probe_wake_or_brain(ep: LaptopEndpoint, timeout: float = 1.5) -> bool:
    for url in (f"{ep.wake_url}/status", f"{ep.brain_url}/api/state"):
        try:
            r = requests.get(url, timeout=timeout)
            if r.ok:
                return True
        except Exception:
            pass
    return False


def wake_laptop(ep: LaptopEndpoint, *, timeout_s: float = 10.0) -> dict[str, Any]:
    """Request the laptop Wake Daemon to boot the RTX 4080 stack (or confirm already ready)."""
    try:
        r = requests.post(f"{ep.wake_url}/wake", json={}, timeout=timeout_s)
        if r.ok:
            return r.json()
    except Exception:
        pass
    # If Wake Daemon is not running, check whether launch_rocky.bat already booted the stack.
    try:
        r = requests.get(f"{ep.brain_url}/api/state", timeout=2.0)
        if r.ok:
            return {"state": "ready", "stage": "ready", "detail": "Connected directly to running brain"}
    except Exception:
        pass
    return {"state": "error", "stage": "error", "detail": f"Laptop unreachable at {ep.host}"}


def poll_wake_status(ep: LaptopEndpoint, *, timeout_s: float = 3.0) -> dict[str, Any]:
    try:
        r = requests.get(f"{ep.wake_url}/status", timeout=timeout_s)
        if r.ok:
            return r.json()
    except Exception:
        pass
    try:
        r = requests.get(f"{ep.brain_url}/api/state", timeout=2.0)
        if r.ok:
            return {"state": "ready", "stage": "ready", "detail": "Connected directly to running brain"}
    except Exception:
        pass
    return {"state": "waking", "stage": "waiting", "detail": f"Waiting for laptop at {ep.host}..."}


def resolve_brain_url(ep: LaptopEndpoint, *, timeout_s: float = 1.5) -> str:
    """Return direct brain URL (:8090) if reachable from robot, else Wake Daemon proxy (:8091/brain)."""
    for candidate in (ep.brain_url, f"{ep.wake_url}/brain"):
        try:
            r = requests.get(f"{candidate}/api/state", timeout=timeout_s)
            if r.ok:
                return candidate
        except Exception:
            continue
    return ep.brain_url


def sleep_laptop(ep: LaptopEndpoint, *, timeout_s: float = 12.0) -> None:
    """Tell the laptop Wake Daemon to end the session and release all GPU VRAM."""
    try:
        requests.post(f"{ep.wake_url}/sleep", json={}, timeout=timeout_s)
    except Exception as exc:
        log.debug("sleep_laptop ignored: %s", exc)
