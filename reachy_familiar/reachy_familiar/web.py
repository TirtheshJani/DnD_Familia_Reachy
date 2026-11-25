"""Robot-hosted Web UI (:8043): live boot screen while waking, then transparent Companion UI proxy."""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import requests

log = logging.getLogger("reachy_familiar.web")

BOOT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rocky // Waking Familiar</title>
<style>
  body {
    margin: 0; min-height: 100vh; display: grid; place-items: center;
    background: radial-gradient(circle at top, #1b1f2e, #0c0e14);
    color: #f3ead3; font-family: system-ui, -apple-system, sans-serif;
  }
  .card {
    width: min(92vw, 440px); padding: 28px; border-radius: 10px;
    border: 1px solid #b5923b; background: rgba(18, 22, 32, 0.94);
    box-shadow: 0 20px 50px rgba(0,0,0,0.6); text-align: center;
  }
  .kicker { font-size: 0.72rem; letter-spacing: 0.2em; color: #d4af37; margin: 0 0 8px; }
  h1 { margin: 0 0 12px; font-size: 1.5rem; letter-spacing: 0.08em; }
  .stage { color: #f0d78c; font-weight: 600; margin: 14px 0 8px; font-size: 1.05rem; }
  .detail { color: #a7b0c2; font-size: 0.9rem; margin: 0 0 18px; min-height: 1.4em; }
  .bar { height: 6px; border-radius: 99px; background: #222838; overflow: hidden; }
  .fill { height: 100%; width: 25%; background: linear-gradient(90deg, #9d7b25, #f0d78c); transition: width 0.5s ease; }
</style>
</head>
<body>
  <div class="card">
    <p class="kicker">SHADOWS OVER NEPHRIA</p>
    <h1>ROCKY // FAMILIAR</h1>
    <div id="stage" class="stage">Discovering Zephyrus RTX 4080...</div>
    <div id="detail" class="detail">Starting local brain stack on your laptop</div>
    <div class="bar"><div id="fill" class="fill"></div></div>
  </div>
  <script>
    const PCT = { discovering: 15, starting: 32, brain: 60, llama: 75, s2s: 90, audio_tap: 96, ready: 100 };
    async function poll() {
      try {
        const r = await fetch('/_familiar/boot-status', { cache: 'no-store' });
        const d = await r.json();
        document.getElementById('stage').textContent = (d.stage || d.state || 'waking').toUpperCase();
        document.getElementById('detail').textContent = d.detail || '';
        document.getElementById('fill').style.width = (PCT[d.stage] || 35) + '%';
        if (d.brain_ready || d.ready) { window.location.replace('/'); return; }
      } catch (_) {}
      setTimeout(poll, 1200);
    }
    poll();
  </script>
</body>
</html>
"""


class CompanionProxyServer:
    """Serves the boot screen while waking, then proxies the laptop's Companion UI."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8043,
        status_provider: Callable[[], dict[str, Any]],
        action_handlers: dict[str, Callable[[], dict[str, Any]]] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.status_provider = status_provider
        self.action_handlers = action_handlers or {}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._http = requests.Session()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        status_fn = self.status_provider
        actions = self.action_handlers
        http_sess = self._http

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                log.debug("web: " + fmt, *args)

            def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
                    pass

            def _proxy(self, method: str) -> None:
                st = status_fn()
                clean_path = self.path.split("?", 1)[0]
                if clean_path == "/_familiar/boot-status":
                    self._send_bytes(200, json.dumps(st).encode("utf-8"), "application/json; charset=utf-8")
                    return
                if method == "POST" and clean_path in ("/_familiar/speak-test", "/_familiar/reconnect-audio"):
                    clen = int(self.headers.get("Content-Length") or 0)
                    if clen > 0:
                        self.rfile.read(clen)
                    key = "speak_test" if clean_path.endswith("speak-test") else "reconnect_audio"
                    fn = actions.get(key)
                    if fn is None:
                        self._send_bytes(
                            503,
                            json.dumps({"ok": False, "error": f"{key} unavailable"}).encode("utf-8"),
                            "application/json; charset=utf-8",
                        )
                        return
                    try:
                        res = fn()
                        code = 200 if res.get("ok", True) else 503
                        self._send_bytes(code, json.dumps(res).encode("utf-8"), "application/json; charset=utf-8")
                    except Exception as exc:
                        self._send_bytes(
                            500,
                            json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"),
                            "application/json; charset=utf-8",
                        )
                    return

                brain_url = st.get("brain_url")
                ui_ready = bool(st.get("brain_ready") or st.get("ready"))
                if not ui_ready or not brain_url:
                    # Serve the auto-refreshing Art Deco boot page for ANY browser route/query string.
                    if method == "GET" and not clean_path.startswith(("/api/", "/_familiar/")):
                        self._send_bytes(200, BOOT_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    else:
                        self._send_bytes(503, json.dumps(st).encode("utf-8"), "application/json; charset=utf-8")
                    return

                clen = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(clen) if clen > 0 else None
                fwd_headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in ("host", "connection", "content-length", "accept-encoding")
                }
                target = f"{brain_url.rstrip('/')}{self.path}"
                is_sse = self.path.split("?", 1)[0] == "/api/events"
                headers_sent = False
                try:
                    with http_sess.request(
                        method,
                        target,
                        headers=fwd_headers,
                        data=body,
                        stream=is_sse,
                        timeout=(3.0, None if is_sse else 30.0),
                    ) as resp:
                        self.send_response(resp.status_code)
                        ct = resp.headers.get("Content-Type", "application/octet-stream")
                        self.send_header("Content-Type", ct)
                        if is_sse:
                            self.send_header("Cache-Control", "no-cache")
                            self.send_header("Connection", "close")
                            self.end_headers()
                            headers_sent = True
                            for chunk in resp.iter_content(chunk_size=1024):
                                if not chunk:
                                    continue
                                self.wfile.write(chunk)
                                self.wfile.flush()
                        else:
                            content = resp.content
                            self.send_header("Content-Length", str(len(content)))
                            self.send_header("Connection", "close")
                            self.end_headers()
                            headers_sent = True
                            self.wfile.write(content)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    pass
                except Exception as exc:
                    if not headers_sent:
                        self._send_bytes(
                            502,
                            json.dumps({"error": f"Companion proxy error: {exc}"}).encode("utf-8"),
                            "application/json; charset=utf-8",
                        )

            def do_GET(self) -> None:
                self._proxy("GET")

            def do_POST(self) -> None:
                self._proxy("POST")

            def do_PUT(self) -> None:
                self._proxy("PUT")

            def do_DELETE(self) -> None:
                self._proxy("DELETE")

        return Handler

    def start(self) -> None:
        self._server = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="familiar-web")
        self._thread.start()
        log.info("reachy_familiar Web UI listening on http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._http.close()
