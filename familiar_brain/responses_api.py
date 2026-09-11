from __future__ import annotations

import copy
import json
import time
import uuid

IDLE_TOOL = "idle_do_nothing"


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


def wants_stream(req: dict) -> bool:
    """True only when the client explicitly asked for SSE.

    speech-to-speech's ResponsesApiModelHandler.warmup() and its compaction helper
    both call responses.create() without `stream`, so an absent flag must mean a
    plain JSON Response body, not an event stream.
    """
    return bool(req.get("stream"))


def sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def _idle_item() -> dict:
    # call_id must be client-supplied and prefixed 'call_': speech_to_speech's
    # Chat.add_item rejects a function_call without one.
    return {"id": f"fc_{uuid.uuid4().hex[:8]}", "type": "function_call", "status": "completed",
            "call_id": f"call_{uuid.uuid4().hex[:12]}", "name": IDLE_TOOL, "arguments": "{}"}


def idle_response_events(*, response_id: str | None = None) -> list[dict]:
    rid = response_id or f"resp_{uuid.uuid4().hex[:12]}"
    item = _idle_item()
    return [
        {"type": "response.created", "response": {"id": rid, "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**item, "status": "in_progress", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": item["id"], "delta": "{}"},
        {"type": "response.function_call_arguments.done", "output_index": 0, "item_id": item["id"], "arguments": "{}"},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": {"id": rid, "status": "completed", "output": [item]}},
    ]


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
