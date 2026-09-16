from __future__ import annotations

import copy
import json
import time
import uuid

IDLE_TOOL = "idle_do_nothing"
CAMERA_TOOL = "camera"
# The conversation app opens a session by asking the model for a greeting with this
# exact prompt (prompts.py DEFAULT_GREETING_PROMPT). It is not a table utterance.
GREETING_MARKER = "Start the conversation now"


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("input_text", "text", "output_text"))
    return ""


def latest_user_text(req: dict) -> str:
    for item in reversed(req.get("input") or []):
        if isinstance(item, dict) and item.get("role") == "user":
            return _text_of(item.get("content")).strip()
    return ""


def function_call_outputs(req: dict) -> list[tuple[str, str]]:
    return [(i["call_id"], str(i.get("output", ""))) for i in (req.get("input") or [])
            if isinstance(i, dict) and i.get("type") == "function_call_output"]


def continuation_kind(req: dict) -> str | None:
    """Classify a request that is not a fresh utterance from the table.

    None:       an ordinary utterance; run the gate. A user message that carries text
                alongside an image is still an utterance.
    "sight":    the client just attached a camera frame, or returned the camera tool's
                result; the next generation must see it, so the gate is bypassed.
    "echo":     the client returned the output of another robot-side tool (the app
                requests a response after idle_do_nothing and camera, which keep
                needs_response=True). Answer with nothing; it cannot loop.
    "greeting": the app's session-open greeting prompt.
    Shapes from huggingface_realtime.py (app 1.0.1) and docs/protocol-notes.md 4 and 6.
    """
    items = req.get("input")
    # The Responses API also accepts a bare string as the whole input; that is an
    # utterance, never a continuation.
    if not isinstance(items, list) or not items:
        return None
    last = items[-1]
    if not isinstance(last, dict):
        return None
    if last.get("role") == "user":
        content = last.get("content")
        if isinstance(content, list):
            kinds = {p.get("type") for p in content if isinstance(p, dict)}
            if "input_image" in kinds and not kinds & {"input_text", "text"}:
                return "sight"
        return "greeting" if GREETING_MARKER in _text_of(content) else None
    if last.get("type") == "function_call_output":
        # A result whose call is missing (the client may have trimmed history) is an
        # echo too: a tool result is never something the table said.
        names = {i.get("call_id"): i.get("name") for i in items
                 if isinstance(i, dict) and i.get("type") == "function_call"}
        return "sight" if names.get(last.get("call_id")) == CAMERA_TOOL else "echo"
    return None


def wants_stream(req: dict) -> bool:
    """True only when the client explicitly asked for SSE.

    speech-to-speech's ResponsesApiModelHandler.warmup() and its compaction helper
    both call responses.create() without `stream`, so an absent flag must mean a
    plain JSON Response body, not an event stream.
    """
    return bool(req.get("stream"))


def sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def call_item(name: str, args: dict | None = None) -> dict:
    """A completed function_call item for a tool the robot owns.

    call_id must be client-supplied and prefixed 'call_': speech_to_speech's
    Chat.add_item rejects a function_call without one (docs/protocol-notes.md 3).
    """
    return {"id": f"fc_{uuid.uuid4().hex[:8]}", "type": "function_call", "status": "completed",
            "call_id": f"call_{uuid.uuid4().hex[:12]}", "name": name,
            "arguments": json.dumps(args or {}, separators=(",", ":"))}


def _idle_item() -> dict:
    return call_item(IDLE_TOOL)


def idle_items() -> list[dict]:
    """The one-item preface for a turn that has nothing else to send."""
    return [_idle_item()]


def item_events(items: list[dict], *, start_index: int = 0) -> list[dict]:
    """The added, delta, done, done quartet for each item, without the envelope.

    Used on its own to preface an upstream text stream with gestures: speech-to-speech
    consumes only output_item.done, output_text.delta and completed, so the preface can
    precede the upstream response.created without confusing it.
    """
    out: list[dict] = []
    for i, item in enumerate(items, start=start_index):
        out += [
            {"type": "response.output_item.added", "output_index": i,
             "item": {**item, "status": "in_progress", "arguments": ""}},
            {"type": "response.function_call_arguments.delta", "output_index": i,
             "item_id": item["id"], "delta": item["arguments"]},
            {"type": "response.function_call_arguments.done", "output_index": i,
             "item_id": item["id"], "arguments": item["arguments"]},
            {"type": "response.output_item.done", "output_index": i, "item": item},
        ]
    return out


def tool_call_events(items: list[dict], *, response_id: str | None = None) -> list[dict]:
    """A complete streamed response whose only output is the given tool calls."""
    rid = response_id or f"resp_{uuid.uuid4().hex[:12]}"
    return ([{"type": "response.created", "response": {"id": rid, "status": "in_progress", "output": []}}]
            + item_events(items)
            + [{"type": "response.completed", "response": {"id": rid, "status": "completed", "output": list(items)}}])


def idle_response_events(*, response_id: str | None = None) -> list[dict]:
    return tool_call_events(idle_items(), response_id=response_id)


def empty_response_events(*, response_id: str | None = None) -> list[dict]:
    """A completed response with nothing in it. The answer to a tool-result echo the
    client sends after a robot-side tool that needed no reply; it cannot start a loop
    because it carries no tool call for the client to execute."""
    return tool_call_events([], response_id=response_id)


def response_json(output: list[dict], *, response_id: str | None = None, model: str = "familiar") -> dict:
    """A complete non-streaming Responses API body.

    The openai SDK parses this into a Response model, so the envelope fields matter
    even when the caller only times the round trip.
    """
    return {
        "id": response_id or f"resp_{uuid.uuid4().hex[:12]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def idle_response_json(*, response_id: str | None = None, model: str = "familiar") -> dict:
    return response_json([_idle_item()], response_id=response_id, model=model)


def with_instructions_and_tools(req: dict, *, instructions: str, extra_tools: list[dict]) -> dict:
    out = copy.deepcopy(req)
    out["instructions"] = instructions
    have = {t.get("name") for t in out.get("tools") or []}
    out["tools"] = list(out.get("tools") or []) + [t for t in extra_tools if t.get("name") not in have]
    return out
