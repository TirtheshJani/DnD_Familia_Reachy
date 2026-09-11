"""Transparent logging proxy: speech-to-speech -> this -> llama-server.
Writes the first request body and the full upstream SSE stream to files, then keeps relaying.
Usage: python scripts/capture_protocol.py --upstream http://127.0.0.1:8080 --port 8090

Non-streaming requests are captured separately, because the client's warmup probe is
non-streaming and arrives before any speech. See docs/protocol-notes.md.
"""
import argparse
import json
import sys
from pathlib import Path

import requests
from flask import Flask, Response, request, stream_with_context

app = Flask(__name__)
OUT = Path("tests/fixtures")
OUT.mkdir(parents=True, exist_ok=True)
captured = {"request": False, "stream": False, "json": False}


@app.route("/v1/<path:sub>", methods=["POST", "GET"])
def relay(sub):
    url = f"{app.config['UPSTREAM']}/v1/{sub}"
    body = request.get_data()
    if request.method == "POST" and not captured["request"]:
        (OUT / "captured_request.json").write_bytes(body)
        captured["request"] = True
        print(f"[capture] wrote request for /v1/{sub}", file=sys.stderr)
    try:
        streamed = bool((json.loads(body or b"{}") or {}).get("stream"))
    except (ValueError, TypeError):
        streamed = False
    if request.method == "POST" and streamed and not captured.get("stream_request"):
        (OUT / "captured_request_stream.json").write_bytes(body)
        captured["stream_request"] = True
        print("[capture] wrote first streaming request", file=sys.stderr)
    up = requests.request(request.method, url, data=body,
                          headers={"Content-Type": request.content_type or "application/json"},
                          stream=True, timeout=120)
    ctype = up.headers.get("Content-Type", "application/json")

    def gen():
        chunks = []
        for chunk in up.iter_content(chunk_size=None):
            if chunk:
                chunks.append(chunk)
                yield chunk
        raw = b"".join(chunks)
        if "text/event-stream" in ctype and not captured["stream"]:
            (OUT / "captured_stream.sse").write_bytes(raw)
            captured["stream"] = True
            print("[capture] wrote SSE stream", file=sys.stderr)
        elif "text/event-stream" not in ctype and not captured["json"]:
            (OUT / "captured_response.json").write_bytes(raw)
            captured["json"] = True
            print("[capture] wrote non-streaming response body", file=sys.stderr)

    return Response(stream_with_context(gen()), status=up.status_code, content_type=ctype)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default="http://127.0.0.1:8080")
    p.add_argument("--port", type=int, default=8090)
    a = p.parse_args()
    app.config["UPSTREAM"] = a.upstream
    app.run(host="127.0.0.1", port=a.port, threaded=True)
