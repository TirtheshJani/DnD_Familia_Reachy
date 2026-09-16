from __future__ import annotations

import copy
import json
import logging
import uuid
from typing import Iterator

import requests

from familiar_brain import tools
from familiar_brain.responses_api import CAMERA_TOOL, sse

MAX_TOOL_ROUNDS = 3

log = logging.getLogger("familiar.upstream")


class Upstream:
    def __init__(self, *, base_url: str, generate_slot: int, brain_tool_names: set[str], timeout_s: float = 60.0,
                 reasoning_effort: str | None = "none"):
        self.base_url = base_url.rstrip("/")
        self.generate_slot = generate_slot
        self.brain_tool_names = brain_tool_names
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort

    def _with_defaults(self, body: dict, *, stream: bool) -> dict:
        """Gemma 4 is a reasoning model. Left alone it spends the output budget on a
        reasoning item and then answers in prose rather than emitting the function_call,
        so the dice never actually roll and the turn misses the latency ceiling. A
        caller that sets reasoning itself, via --responses_api_reasoning_effort, wins.
        """
        out = {**body, "stream": stream, "id_slot": self.generate_slot}
        if self.reasoning_effort is not None:
            out.setdefault("reasoning", {"effort": self.reasoning_effort})
        return out

    def _post(self, body: dict) -> requests.Response:
        r = requests.post(f"{self.base_url}/responses", json=self._with_defaults(body, stream=True),
                          stream=True, timeout=self.timeout_s)
        r.raise_for_status()
        # llama-server sends text/event-stream with no charset, for which requests assumes
        # Latin-1, so an apostrophe in a reply reached text-to-speech as three characters.
        r.encoding = "utf-8"
        return r

    def complete(self, req: dict) -> dict:
        """Non-streaming passthrough, returning the upstream Response body verbatim.

        Used for the client's warmup probe and history compaction. Neither declares
        tools, so no brain-side tool round trip is attempted on this path.
        """
        body = self._with_defaults(copy.deepcopy(req), stream=False)
        r = requests.post(f"{self.base_url}/responses", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _iter_events(resp: requests.Response) -> Iterator[dict]:
        buf = ""
        for line in resp.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line == "":
                if buf:
                    for l in buf.splitlines():
                        if l.startswith("data: "):
                            yield json.loads(l[6:])
                    buf = ""
            else:
                buf += line + "\n"

    def stream(self, req: dict, *, must_speak: bool = False) -> Iterator[str]:
        """Yield SSE frames for the client. Brain tools are executed here and hidden from the client.

        must_speak: the turn was addressed to the familiar, so it must not end without
        words. When the model's turn completes with no text and no camera call (whose
        words come on the sight continuation), its response.completed is held back and
        the model is asked once more with no tools offered, keeping any brain tool
        results already in the body. Measured on the live model on 2026-09-16: with no
        tools, 8 of 8 addressed lines got words, and 3 of 3 reported a roll result that
        was already in the history. Asking again with tool_choice "none" instead made
        Gemma speak call syntax aloud ("<|tool_call>call:move_head...").
        """
        body = copy.deepcopy(req)
        last_response_id: str | None = None
        answered = asked_again = False
        rounds = 0
        while rounds <= MAX_TOOL_ROUNDS:
            pending: list[dict] = []
            ask_again = False
            # Argument frames for a brain tool reference their item by item_id and carry
            # no nested item, so the item-shaped check below cannot see them. Collect the
            # ids as the items are announced and suppress anything pointing at them.
            hidden_item_ids: set[str] = set()
            for ev in self._iter_events(self._post(body)):
                etype = ev["type"]
                item = ev.get("item") or {}
                rid = (ev.get("response") or {}).get("id")
                if rid:
                    last_response_id = rid
                if item.get("type") == "function_call" and item.get("name") in self.brain_tool_names:
                    if item.get("id"):
                        hidden_item_ids.add(item["id"])
                    if etype == "response.output_item.done":
                        pending.append(item)
                    continue  # hide brain tool events from the client
                if ev.get("item_id") in hidden_item_ids:
                    continue
                if etype == "response.completed":
                    if pending:
                        break  # do not forward completion yet; we have a round trip to do
                    if must_speak and not answered and not asked_again:
                        ask_again = True
                        break  # hold the completion back; the client ends the turn on it
                    yield sse(ev)
                    return
                if etype == "response.output_text.delta" and str(ev.get("delta") or "").strip():
                    answered = True
                elif (etype == "response.output_item.done" and item.get("type") == "function_call"
                      and item.get("name") == CAMERA_TOOL):
                    answered = True
                yield sse(ev)
            if ask_again:
                log.warning("addressed turn came back without words; asking again with no tools")
                asked_again = True
                body = {k: v for k, v in body.items() if k not in ("tools", "tool_choice")}
                continue
            if not pending:
                return
            rounds += 1
            for item in pending:
                try:
                    args = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                output = tools.run(item["name"], args)
                body["input"] = list(body.get("input") or []) + [
                    {"type": "function_call", "call_id": item["call_id"], "name": item["name"], "arguments": item.get("arguments") or "{}"},
                    {"type": "function_call_output", "call_id": item["call_id"], "output": output},
                ]
        # The round budget is spent and the model is still asking for tools. Close the
        # stream cleanly rather than leaving the client on a truncated response.
        log.warning("tool round budget of %d exhausted; terminating the stream", MAX_TOOL_ROUNDS)
        yield sse({
            "type": "response.completed",
            "response": {"id": last_response_id or f"resp_{uuid.uuid4().hex[:12]}",
                         "status": "completed", "output": []},
        })
