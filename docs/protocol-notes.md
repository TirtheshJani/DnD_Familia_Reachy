# Responses API contract that speech-to-speech expects

Source of truth: the installed package, not the design document. Established by
reading `speech-to-speech` 1.0.0 inside WSL2 at
`~/.venvs/s2s/lib/python3.12/site-packages/speech_to_speech/`.

Plan Task 2 asks for this to come from a captured live exchange. The source read
came first because it is authoritative about which events are consumed and which
are discarded, which a single capture cannot show. The live capture is recorded in
the "Live capture" section below.

## 1. The client is the official openai SDK

`LLM/responses_api_language_model.py` builds requests with
`self.client.responses.create(model=..., input=..., stream=..., tools=...)` and
iterates the result as an `openai.Stream`. Consequences:

- The brain is talking to a typed Pydantic client, not to a hand-rolled SSE reader.
  Event payloads must carry the fields the SDK's models expect, and the `type`
  discriminator must match exactly.
- `extra_body` is merged into the request body by the SDK, so unknown top-level
  request keys can appear.

## 2. Only three stream events are consumed

`ResponsesApiModelHandler._iter_stream_events` (lines 178 to 193) dispatches on
exactly three types. Everything else is silently dropped.

| Event type | SDK model | What the handler takes from it |
|---|---|---|
| `response.output_text.delta` | `ResponseTextDeltaEvent` | `.delta`, emitted as spoken text |
| `response.output_item.done` | `ResponseOutputItemDoneEvent` | `.item`; a `function_call` item becomes a ToolCall, a `message` item becomes an AssistantMessage |
| `response.completed` | `ResponseCompletedEvent` | `.response.usage` only |

`response.created`, `response.output_item.added`,
`response.function_call_arguments.delta`, `response.function_call_arguments.done`
and `response.content_part.*` are **not** consumed on this path. The brain still
emits the full sequence, because emitting a well-formed stream costs nothing and
keeps the stream valid for any other client.

The other `response.*` literals found elsewhere in the package belong to the
robot-facing OpenAI Realtime websocket API (`api/openai_realtime/`), not to this
upstream contract. Do not conflate the two.

## 3. function_call items must carry a client-supplied call_id prefixed `call_`

`LLM/chat.py` around line 1207:

```
# call_id on function_call items must be client-supplied: it is referenced later by
# function_call_output items, so we cannot silently generate one here.
...
raise ChatItemError("function_call item is missing a call_id. The call_id should start with 'call_'.")
```

So `idle_response_events` and any forwarded tool call must set
`call_id="call_..."`. A missing or differently prefixed id raises inside the client.

Note the handler then **overwrites** both ids with its own
(`item.call_id = _generate_id("call")`, `item.id = _generate_id("fc")`, lines 185
to 186). The brain therefore cannot rely on its own `call_id` surviving a round
trip through speech-to-speech, which is a further reason brain-side tools are
executed in the brain rather than handed out.

## 4. The client owns the tool round trip for robot-side tools

`LLM/base_openai_compatible_language_model.py` lines 543 to 546:

```
The function_call must already exist in the conversation by the time the
client returns its function_call_output; otherwise a fast client ...
rejected ("No function_call with call_id ... found")
```

So for tools the robot owns (`play_emotion`, `move_head`, `head_tracking`,
`camera`, `idle_do_nothing`), the flow is: brain streams a `function_call` to the
client, the app executes it, and the client sends a **new** request whose `input`
contains a `function_call_output` item. The brain relays that upstream unchanged.

Brain-side tools (`roll`, `initiative`, `note`) are the opposite case. The robot
has no implementation for them, so the brain must intercept those `function_call`
items, execute them locally, and round trip to llama-server itself without the
client ever seeing them. This is what Task 11 implements, and the capture confirms
the plan's assumption for that path was the right one.

## 5. Non-streaming is required, not optional

`warmup()` (line 68) runs at pipeline start:

```python
self.client.with_options(max_retries=WARMUP_MAX_RETRIES).responses.create(
    model=self.model_name,
    input=[...system..., ...user "Hello"...],
    timeout=self.request_timeout,
)
```

There is no `stream=` argument, so this is a **non-streaming** request expecting a
JSON `Response` body. `_build_compaction_generate_fn` does the same and reads
`response.output_text`.

The plan's `/v1/responses` returns `text/event-stream` unconditionally, which would
fail at startup. The brain therefore branches on `wants_stream(req)` and returns a
complete Response object (`response_json`) when the flag is absent or false.

## 6. Input item shapes

From `warmup()` and `Chat._to_responses_api_chat_locked`:

- Messages: `{"type": "message", "role": "system"|"user", "content": [{"type": "input_text", "text": ...}]}`.
  Assistant turns use `output_text` parts. `latest_user_text` handles both the
  parts list and a bare string.
- Tool call: `{"type": "function_call", "call_id": ..., "name": ..., "arguments": <json string>, "id"?, "status"?}`.
- Tool result: `{"type": "function_call_output", "call_id": ..., "output": ..., "id"?, "status"?}`.

## 7. The installed CLI differs from the design document

The design document's launch line is from the May and June 2026 blog posts and does
not match 1.0.0. Verified differences:

- The entry point is subcommand based: `serve`, `talk`, `local`. There is **no
  `--mode realtime` flag**; `serve` is the realtime server.
- Still present and correct: `--stt parakeet-tdt`, `--tts qwen3`,
  `--llm_backend responses-api`, `--responses_api_base_url`,
  `--responses_api_api_key`, `--responses_api_stream`, `--model_name`,
  `--enable_live_transcription`, `--thresh`, `--chat_size`, `--host`, `--port`,
  `--qwen3_tts_model_name`, `--qwen3_tts_speaker`, `--qwen3_tts_backend`.
- `--stt`, `--tts` and `--llm_backend` already default to the values this project
  wants (`parakeet-tdt`, `qwen3`, `responses-api`).
- New since the blog and worth knowing: `--smart_turn` turn detection,
  `--num_pipelines`, `--responses_api_disable_thinking`, `--compact_history`,
  `--log_transcripts` (off by default, which suits a table full of other people).

See `docs/versions.md` for the exact working command.

## Live capture

Pending. Run `scripts/capture_protocol.py` between speech-to-speech and
llama-server, speak one line at the robot, then record here what the captured
request and stream contain, and reconcile any difference against sections 2 to 6.
