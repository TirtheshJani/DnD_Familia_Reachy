"""In-process fake of llama-server for tests. Serves /v1/chat/completions (non-stream)
and /v1/responses (SSE stream). Scripted replies via queues."""
from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLLM:
    def __init__(self):
        self.chat_replies: deque[str] = deque()
        self.responses_events: deque[list[dict]] = deque()
        self.requests: list[dict] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()

    def _handler(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append({"path": self.path, "body": body})
                if self.path.endswith("/chat/completions"):
                    text = outer.chat_replies.popleft() if outer.chat_replies else "IGNORE"
                    out = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
                elif self.path.endswith("/responses"):
                    events = outer.responses_events.popleft() if outer.responses_events else [
                        {"type": "response.output_text.delta", "delta": "hello"},
                        {"type": "response.completed", "response": {"id": "r1", "output": []}},
                    ]
                    self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                    for ev in events:
                        self.wfile.write(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()
                else:
                    self.send_response(404); self.end_headers()
        return H
