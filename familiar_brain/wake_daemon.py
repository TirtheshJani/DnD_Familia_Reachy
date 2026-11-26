"""Lightweight 0-VRAM Windows Wake Daemon for Reachy Mini's native reachy_familiar app.

Runs at Windows logon (~15 MB RAM, 0 MiB GPU VRAM when idle):
  - UDP :8092 responds to REACHY_FAMILIAR_DISCOVER LAN broadcasts with the laptop's
    current DHCP LAN IP and ports (8091 wake, 8090 brain/UI, 8765 realtime WS).
  - HTTP :8091 serves GET /status, POST /wake, and POST /sleep to boot and shut down
    the WSL2 RTX 4080 stack (llama-server, familiar-brain, speech-to-speech, audio-tap)
    on demand when you press Play or Stop in the Reachy Mini Desktop App or Dashboard.
"""
from __future__ import annotations

import argparse
import http.client
import json
import logging
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, TextIO

log = logging.getLogger("familiar.wake_daemon")

DISCOVER_MAGIC = b"REACHY_FAMILIAR_DISCOVER"
DEFAULT_WAKE_PORT = 8091
DEFAULT_DISCOVER_PORT = 8092
DEFAULT_BRAIN_PORT = 8090
DEFAULT_WS_PORT = 8765
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(repo_root: Path) -> RotatingFileHandler:
    """Log to data/logs/wake_daemon.log as well: under pythonw.exe there is no stderr to read."""
    log_dir = Path(repo_root) / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "wake_daemon.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return handler


def detect_lan_ip(peer_ip: str = "10.0.0.100") -> str:
    """Return the local IPv4 address used to reach peer_ip (no packet transmitted)."""
    for target in (peer_ip, "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((target, 8000))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
        except OSError:
            continue
    return "127.0.0.1"


def windows_to_wsl_path(repo_root: Path) -> str:
    resolved = str(repo_root.resolve())
    if len(resolved) >= 2 and resolved[1] == ":":
        drive = resolved[0].lower()
        rest = resolved[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return resolved.replace("\\", "/")


def _http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _http_post_quiet(url: str, timeout: float = 3.0) -> None:
    try:
        req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except Exception:
        pass


class StackManager:
    """Thread-safe lifecycle state machine for waking and sleeping the WSL2 GPU stack."""

    def __init__(
        self,
        *,
        repo_root: Path,
        distro: str = "Ubuntu-24.04",
        brain_port: int = DEFAULT_BRAIN_PORT,
        ws_port: int = DEFAULT_WS_PORT,
        wake_port: int = DEFAULT_WAKE_PORT,
        spawn_fn: Callable[[], subprocess.Popen[str]] | None = None,
        stop_fn: Callable[[subprocess.Popen[str] | None], None] | None = None,
        probe_fn: Callable[[], bool] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.distro = distro
        self.brain_port = brain_port
        self.ws_port = ws_port
        self.wake_port = wake_port
        self._spawn_fn = spawn_fn
        self._stop_fn = stop_fn
        self._probe_fn = probe_fn

        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._stack_log_path: Path | None = None
        self._state = "sleeping"
        self._stage = "idle"
        self._detail = "GPU stack sleeping (0 MiB VRAM)"
        self._brain_ready = False
        self._voice_ready = False
        self._updated_at = time.time()

    def _is_brain_live(self) -> bool:
        if self._probe_fn is not None:
            return bool(self._probe_fn())
        return _http_ok(f"http://127.0.0.1:{self.brain_port}/api/state")

    def _is_stack_live(self) -> bool:
        if self._probe_fn is not None:
            return bool(self._probe_fn())
        return _http_ok(f"http://127.0.0.1:{self.brain_port}/api/state") and _http_ok(
            f"http://127.0.0.1:{self.ws_port}/v1/pool"
        )

    def _default_spawn(self) -> subprocess.Popen[str]:
        wsl_repo = windows_to_wsl_path(self.repo_root)
        # The daemon names the run dir so it can write the stack's stdout to stack.log there;
        # start_all.sh puts its child logs in the RUN_DIR it is given.
        run_name = time.strftime("%Y%m%d-%H%M%S")
        run_dir = self.repo_root / "data" / "logs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._stack_log_path = run_dir / "stack.log"
        bash_cmd = (
            f"unset SSL_CERT_FILE; cd '{wsl_repo}' && "
            f"RUN_DIR='data/logs/{run_name}' ./scripts/start_all.sh --native-robot --auto-session"
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(
            ["wsl", "-d", self.distro, "-e", "bash", "-lc", bash_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )

    def _default_stop(self, proc: subprocess.Popen[str] | None) -> None:
        _http_post_quiet(f"http://127.0.0.1:{self.brain_port}/api/session/end", timeout=3.0)
        time.sleep(0.5)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(
                [
                    "wsl",
                    "-d",
                    self.distro,
                    "-e",
                    "bash",
                    "-lc",
                    "pkill -TERM -f 'scripts/start_all.sh|llama-server|familiar-brain|scripts/run_s2s.py|familiar_brain.audio_tap' 2>/dev/null || true",
                ],
                timeout=10,
                check=False,
                creationflags=creationflags,
            )
        except Exception:
            pass

    def status(self, peer_ip: str = "10.0.0.100") -> dict[str, Any]:
        live_now = False
        with self._lock:
            st = self._state
            proc = self._proc
        if st == "sleeping":
            live_now = self._is_stack_live()
            if live_now:
                with self._lock:
                    if self._state == "sleeping":
                        self._state = "ready"
                        self._stage = "ready"
                        self._detail = "Stack already running"
                        self._brain_ready = True
                        self._voice_ready = True
                        self._updated_at = time.time()
        elif st == "waking":
            # Detect brain readiness live even before stdout flushes so UI opens at ~15s.
            if not self._brain_ready and self._is_brain_live():
                with self._lock:
                    if self._state == "waking":
                        self._brain_ready = True
                        if self._stage in ("idle", "starting"):
                            self._stage = "brain"
                            self._detail = "Companion UI ready; voice engine warming up..."
                            self._updated_at = time.time()
        elif st == "ready" and proc is not None and proc.poll() is not None:
            if not self._is_stack_live():
                with self._lock:
                    self._state = "sleeping"
                    self._stage = "idle"
                    self._detail = "Stack exited"
                    self._brain_ready = False
                    self._voice_ready = False
                    self._proc = None
                    self._updated_at = time.time()
        with self._lock:
            return {
                "service": "reachy_familiar_wake",
                "state": self._state,
                "stage": self._stage,
                "detail": self._detail,
                "brain_ready": self._brain_ready,
                "voice_ready": self._voice_ready,
                "host": detect_lan_ip(peer_ip),
                "wake_port": self.wake_port,
                "brain_port": self.brain_port,
                "ws_port": self.ws_port,
                "updated_at": self._updated_at,
            }

    def wake(self) -> dict[str, Any]:
        with self._lock:
            if self._state in ("waking", "ready"):
                return self._status_locked()
            if self._is_stack_live():
                self._state = "ready"
                self._stage = "ready"
                self._detail = "Stack already running"
                self._brain_ready = True
                self._voice_ready = True
                self._updated_at = time.time()
                return self._status_locked()
            self._state = "waking"
            self._stage = "starting"
            self._detail = "Booting RTX 4080 stack in WSL2..."
            self._brain_ready = False
            self._voice_ready = False
            self._updated_at = time.time()
            try:
                proc = (self._spawn_fn or self._default_spawn)()
                self._proc = proc
                stack_log_path = self._stack_log_path
            except Exception as exc:
                log.warning("failed to start stack: %s", exc)
                self._state = "error"
                self._stage = "error"
                self._detail = f"Failed to start stack: {exc}"
                self._updated_at = time.time()
                return self._status_locked()
        threading.Thread(
            target=self._watch_boot, args=(proc, stack_log_path), daemon=True, name="wake-watch"
        ).start()
        return self.status()

    def _watch_boot(self, proc: subprocess.Popen[str], stack_log_path: Path | None = None) -> None:
        stage_markers = (
            ("familiar-brain ready", "brain", "Companion UI ready; loading Gemma 4 & voice..."),
            ("llama-server ready", "llama", "Gemma 4 E4B loaded; warming up voice..."),
            ("speech-to-speech ready", "s2s", "Parakeet STT & Kokoro TTS ready"),
            ("audio-tap ready", "audio_tap", "Speaker & ambient audio tap ready"),
            ("Session auto-started", "ready", "Rocky is awake and listening"),
            ("Skipping robot connection", "ready", "Rocky is awake and listening"),
        )
        stack_log = self._open_stack_log(stack_log_path)
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    if stack_log is not None:
                        try:
                            stack_log.write(line)
                            stack_log.flush()
                        except OSError as exc:
                            log.warning("stopped writing stack.log: %s", exc)
                            stack_log = None
                    clean = line.strip()
                    if not clean:
                        continue
                    log.info("[stack] %s", clean)
                    for marker, stage, detail in stage_markers:
                        if marker in clean:
                            with self._lock:
                                if self._proc is not proc:
                                    return
                                self._stage = stage
                                self._detail = detail
                                if stage in ("brain", "llama", "s2s", "audio_tap", "ready"):
                                    self._brain_ready = True
                                if stage == "ready":
                                    self._voice_ready = True
                                    self._state = "ready"
                                self._updated_at = time.time()
                            break
            rc = proc.wait()
            with self._lock:
                if self._proc is proc:
                    if self._state != "stopping":
                        self._state = "error" if rc not in (0, -15, 143) else "sleeping"
                        self._stage = "error" if self._state == "error" else "idle"
                        self._detail = f"Stack exited (code {rc})" if self._state == "error" else "Stack stopped"
                    else:
                        self._state = "sleeping"
                        self._stage = "idle"
                        self._detail = "GPU stack sleeping (0 MiB VRAM)"
                    self._proc = None
                    self._updated_at = time.time()
        except Exception as exc:
            with self._lock:
                if self._proc is proc:
                    self._state = "error"
                    self._stage = "error"
                    self._detail = f"Boot watcher error: {exc}"
                    self._updated_at = time.time()
        finally:
            if stack_log is not None:
                try:
                    stack_log.close()
                except OSError as exc:
                    log.warning("could not close stack.log: %s", exc)

    @staticmethod
    def _open_stack_log(path: Path | None) -> TextIO | None:
        if path is None:
            return None
        try:
            return open(path, "a", encoding="utf-8")
        except OSError as exc:
            log.warning("cannot open %s; stack output goes to this log only: %s", path, exc)
            return None

    def sleep(self) -> dict[str, Any]:
        with self._lock:
            if self._state == "sleeping" and self._proc is None:
                return self._status_locked()
            self._state = "stopping"
            self._stage = "stopping"
            self._detail = "Ending session and releasing GPU VRAM..."
            self._updated_at = time.time()
            proc = self._proc
        try:
            (self._stop_fn or self._default_stop)(proc)
        finally:
            with self._lock:
                self._proc = None
                self._state = "sleeping"
                self._stage = "idle"
                self._detail = "GPU stack sleeping (0 MiB VRAM)"
                self._brain_ready = False
                self._voice_ready = False
                self._updated_at = time.time()
                return self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        return {
            "service": "reachy_familiar_wake",
            "state": self._state,
            "stage": self._stage,
            "detail": self._detail,
            "brain_ready": self._brain_ready,
            "voice_ready": self._voice_ready,
            "host": detect_lan_ip(),
            "wake_port": self.wake_port,
            "brain_port": self.brain_port,
            "ws_port": self.ws_port,
            "updated_at": self._updated_at,
        }


class WakeDaemonServer:
    """Runs both the HTTP wake/sleep API (:8091) and UDP LAN discovery responder (:8092)."""

    def __init__(
        self,
        manager: StackManager,
        *,
        host: str = "0.0.0.0",
        wake_port: int = DEFAULT_WAKE_PORT,
        discover_port: int = DEFAULT_DISCOVER_PORT,
    ) -> None:
        self.manager = manager
        self.host = host
        self.wake_port = wake_port
        self.discover_port = discover_port
        self._stop = threading.Event()
        self._http: ThreadingHTTPServer | None = None
        self._udp_sock: socket.socket | None = None
        self._threads: list[threading.Thread] = []

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        mgr = self.manager

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                log.debug("http: " + fmt, *args)

            def _send_json(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    pass

            def _proxy_brain(self, method: str) -> None:
                """Forward /brain/* to local WSL2 familiar-brain (127.0.0.1:8090), bypassing Hyper-V LAN blocks."""
                rest = self.path[len("/brain") :]
                if not rest or rest.startswith("?"):
                    rest = "/" + rest
                clen = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(clen) if clen > 0 else None
                fwd_headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in ("host", "connection", "content-length", "accept-encoding")
                }
                is_sse = rest.split("?", 1)[0] == "/api/events"
                conn: http.client.HTTPConnection | None = None
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", mgr.brain_port, timeout=None if is_sse else 30.0)
                    conn.request(method, rest, body=body, headers=fwd_headers)
                    resp = conn.getresponse()
                    self.send_response(resp.status)
                    ct = resp.getheader("Content-Type", "application/octet-stream")
                    self.send_header("Content-Type", ct)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    if is_sse:
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        while True:
                            chunk = resp.read1(1024) if hasattr(resp, "read1") else resp.read(256)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    else:
                        data = resp.read()
                        self.send_header("Content-Length", str(len(data)))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.wfile.write(data)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    pass
                except Exception as exc:
                    self._send_json(502, {"error": f"brain proxy error: {exc}"})
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass

            def do_OPTIONS(self) -> None:
                self._send_json(200, {"ok": True})

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/brain" or path.startswith("/brain/"):
                    self._proxy_brain("GET")
                    return
                if path in ("/status", "/health"):
                    peer = self.client_address[0] if self.client_address else "10.0.0.100"
                    self._send_json(200, mgr.status(peer_ip=peer))
                    return
                self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/brain" or path.startswith("/brain/"):
                    self._proxy_brain("POST")
                    return
                clen = int(self.headers.get("Content-Length") or 0)
                if clen > 0:
                    self.rfile.read(clen)
                if path == "/wake":
                    self._send_json(200, mgr.wake())
                    return
                if path == "/sleep":
                    self._send_json(200, mgr.sleep())
                    return
                self._send_json(404, {"error": "not found"})

            def do_PUT(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/brain" or path.startswith("/brain/"):
                    self._proxy_brain("PUT")
                    return
                self._send_json(404, {"error": "not found"})

            def do_DELETE(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/brain" or path.startswith("/brain/"):
                    self._proxy_brain("DELETE")
                    return
                self._send_json(404, {"error": "not found"})

        return Handler

    def _run_udp(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:
            pass
        sock.bind((self.host, self.discover_port))
        sock.settimeout(0.5)
        self._udp_sock = sock
        self.discover_port = int(sock.getsockname()[1])
        log.info("wake_daemon UDP discovery listening on %s:%d", self.host, self.discover_port)
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            if DISCOVER_MAGIC in data or b'"discover"' in data:
                reply = self.manager.status(peer_ip=addr[0])
                try:
                    sock.sendto(json.dumps(reply).encode("utf-8"), addr)
                except OSError:
                    pass

    def start(self) -> None:
        self._http = ThreadingHTTPServer((self.host, self.wake_port), self._make_handler())
        self.wake_port = int(self._http.server_address[1])
        self.manager.wake_port = self.wake_port
        t_http = threading.Thread(target=self._http.serve_forever, daemon=True, name="wake-http")
        t_udp = threading.Thread(target=self._run_udp, daemon=True, name="wake-udp")
        self._threads = [t_http, t_udp]
        t_http.start()
        t_udp.start()
        log.info("wake_daemon HTTP listening on %s:%d", self.host, self.wake_port)

    def stop(self) -> None:
        self._stop.set()
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
            self._http = None
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except OSError:
                pass
            self._udp_sock = None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="0-VRAM Wake Daemon for Reachy Mini reachy_familiar")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--wake-port", type=int, default=DEFAULT_WAKE_PORT)
    parser.add_argument("--discover-port", type=int, default=DEFAULT_DISCOVER_PORT)
    parser.add_argument("--distro", default="Ubuntu-24.04")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    setup_logging(Path(args.repo))
    mgr = StackManager(repo_root=Path(args.repo), distro=args.distro, wake_port=args.wake_port)
    srv = WakeDaemonServer(mgr, host=args.host, wake_port=args.wake_port, discover_port=args.discover_port)
    srv.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
