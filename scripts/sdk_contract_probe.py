"""Contract check: drive a running familiar-brain with the real openai SDK.

Design spec 4.4 asks for a contract test against a real speech-to-speech request.
This is stronger than a recorded fixture: it uses the same openai SDK version the
installed speech-to-speech uses, and applies the exact isinstance dispatch from
ResponsesApiModelHandler._iter_stream_events. A fixture can drift from the client;
this cannot.

Covers the two paths that do not need llama-server running:
  1. the non-streaming warmup probe that runs at pipeline startup
  2. the IGNORE stream, which is what most table utterances produce

Run with the speech-to-speech venv so the SDK version matches what will be used:
    ~/.venvs/s2s/bin/python scripts/sdk_contract_probe.py [--base-url URL]
Exits non-zero on any contract violation.
"""
from __future__ import annotations

import argparse
import sys

from openai import OpenAI
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)

WARMUP_INPUT = [
    {"type": "message", "role": "system",
     "content": [{"type": "input_text", "text": "You are a helpful assistant"}]},
    {"type": "message", "role": "user",
     "content": [{"type": "input_text", "text": "Hello"}]},
]
UNADDRESSED = "the door creaks open somewhere in the dark"

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def probe_warmup(client: OpenAI) -> None:
    """warmup() omits stream, so the SDK parses a JSON Response. SSE here breaks startup."""
    print("1. non-streaming warmup probe")
    r = client.responses.create(model="familiar", input=WARMUP_INPUT, timeout=30)
    check(type(r).__name__ == "Response", f"SDK parsed a Response object, got {type(r).__name__}")
    check(r.status == "completed", f"status is completed, got {r.status!r}")
    check(isinstance(r.output, list), "output is a list")
    check(r.id.startswith("resp_"), f"id is response-shaped, got {r.id!r}")


def probe_ignore_stream(client: OpenAI) -> None:
    """An unaddressed line must yield one idle tool call and no speech at all."""
    print("2. IGNORE stream, dispatched the way speech-to-speech dispatches it")
    stream = client.responses.create(
        model="familiar",
        input=[{"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": UNADDRESSED}]}],
        stream=True,
        timeout=30,
    )
    tool_calls: list[tuple[str, str, str]] = []
    texts: list[str] = []
    completed = 0
    for ev in stream:
        if isinstance(ev, ResponseTextDeltaEvent):
            texts.append(ev.delta)
        elif isinstance(ev, ResponseOutputItemDoneEvent):
            item = ev.item
            if isinstance(item, ResponseFunctionToolCall):
                tool_calls.append((item.name, item.call_id, item.arguments))
            elif isinstance(item, ResponseOutputMessage):
                texts.append("<message>")
        elif isinstance(ev, ResponseCompletedEvent):
            completed += 1
    check(len(tool_calls) == 1, f"exactly one tool call, got {len(tool_calls)}")
    if tool_calls:
        name, call_id, args = tool_calls[0]
        check(name == "idle_do_nothing", f"tool is idle_do_nothing, got {name!r}")
        check(call_id.startswith("call_"), f"call_id carries the required call_ prefix, got {call_id!r}")
        check(args == "{}", f"arguments are empty json, got {args!r}")
    check(texts == [], f"no speech on an unaddressed line, got {texts!r}")
    check(completed == 1, f"stream completes exactly once, got {completed}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    a = p.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="")
    probe_warmup(client)
    probe_ignore_stream(client)
    print()
    if failures:
        print(f"FAILED: {len(failures)} contract violation(s)")
        return 1
    print("PASS: brain satisfies the speech-to-speech Responses API contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
