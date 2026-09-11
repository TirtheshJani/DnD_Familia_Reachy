# Deviations from Plan 1

Every departure from `plans/2026-09-10-reachy-familiar-plan-1-core.md`, with the reason
and where to check it. Plan 1's own test total was 49; the suite now runs 58, the extra
9 being the tests that cover the defects below.

## A. Defects found in the plan's code

Each was demonstrated by writing the test first and watching it fail against the
plan's code verbatim, then fixed.

### A1. `initiative.run({"action": "show"})` on an empty tracker

The plan ends the show branch with
`("Order: " + ", ".join(...)) or "Order is empty."`. The left operand is never
falsy, because the label itself is a non-empty string, so the fallback is dead code
and an empty order spoke as a dangling `Order: `.

Observed before the fix: `AssertionError: empty order rendered as a dangling label: 'Order: '`.

Fixed in [initiative.py](familiar_brain/tools/initiative.py) by branching on the row
list. Covered by `test_show_on_empty_order_is_readable`.

### A2. Brain-tool argument frames leaked to the robot

`Upstream.stream` hid brain-tool events by inspecting `ev["item"]`. A real
llama-server stream announces the item once (`response.output_item.added`, which
carries `item`) and then emits `response.function_call_arguments.delta` and
`.done`, which identify their item by `item_id` and carry **no** `item` key. Those
two frames therefore fell straight through to the client, so the robot would
receive argument frames for `roll`, a tool it has no implementation for. The plan's
own test did not catch it because its fake only emitted `output_item.done`.

Observed before the fix: `brain tool argument frames leaked to the client:
['response.function_call_arguments.delta', 'response.function_call_arguments.done']`.

Fixed in [upstream.py](familiar_brain/upstream.py) by recording hidden item ids as
items are announced and suppressing anything that references them. Covered by
`test_brain_tool_argument_frames_do_not_leak`.

### A3. Truncated stream when the tool round budget is spent

If the model kept requesting a brain tool for all `MAX_TOOL_ROUNDS + 1` rounds, the
loop fell out having consumed the last `response.completed` without forwarding it,
leaving the client on a stream that simply stopped.

Observed before the fix: `stream ended on response.created`.

Fixed by emitting a terminating `response.completed` after the loop, with a warning
log. Covered by `test_stream_always_terminates_with_completed`.

### A4. `POST /api/session/end` with no active session returned 500

`SessionLog.path` raises `RuntimeError`, which Flask renders as a 500. Now a 409
with an explanatory body. Covered by `test_session_end_without_a_session_is_a_clean_error`.

## B. Contract gaps against the installed speech-to-speech

Found by reading the installed 1.0.0 source rather than the design document. Full
detail in [protocol-notes.md](docs/protocol-notes.md).

### B1. Non-streaming `/v1/responses` is mandatory

`ResponsesApiModelHandler.warmup()` runs at pipeline startup and calls
`responses.create(...)` with **no** `stream` argument, expecting a JSON `Response`
body. The compaction helper does the same and reads `response.output_text`. The
plan's handler returns `text/event-stream` unconditionally, so the pipeline would
have failed to start.

Added `wants_stream`, `response_json` and `idle_response_json` to
[responses_api.py](familiar_brain/responses_api.py), `Upstream.complete` to
[upstream.py](familiar_brain/upstream.py), and a non-streaming branch in
[app.py](familiar_brain/app.py). Covered by three tests including
`test_warmup_shaped_request_gets_a_json_response_not_sse`.

### B2. `call_id` must be client-supplied and prefixed `call_`

`speech_to_speech/LLM/chat.py` raises `ChatItemError` otherwise. The plan already
used a `call_` prefix, so this is a pin rather than a change, asserted now by
`test_idle_events_call_id_is_client_supplied_with_call_prefix` so a future edit
cannot quietly break it.

### B3. The client owns the tool round trip for robot-side tools

Confirmed at `base_openai_compatible_language_model.py:543`. This validates the
plan's Task 11 interception design for brain-side tools: the robot cannot execute
`roll`, `initiative` or `note`, so the brain must run them itself and never expose
them. No code change, question closed.

### B4. The installed CLI differs from the design document

There is no `--mode realtime`. The realtime server is the `serve` subcommand. All
other flags the project needs survive, and `--stt`, `--tts` and `--llm_backend`
already default to the wanted values. `scripts/start_all.sh` uses the real
interface.

## C. Environment adaptations

### C1. Scaffolded into this repo

Plan Task 0 Step 1 creates a fresh repo at `~/code/reachy-familiar`. This repo
already existed for this project, so the package was scaffolded here and `uv init`
skipped in favour of writing `pyproject.toml` directly. The design spec was copied
to `docs/superpowers/specs/` because the plan references it there.

### C2. WSL2 needed reconfiguring, recorded in `%USERPROFILE%\.wslconfig`

Two blocking problems, both outside the plan's scope:

- **Networking.** The robot sits at 10.0.0.100. Under WSL2's default NAT the distro
  had 192.168.94.249, which the robot cannot route to, so nothing bound inside WSL
  was reachable no matter the bind address. The plan's `hostname -I` step would have
  handed the robot an unreachable address. `networkingMode=mirrored` gives the
  distro the host's own LAN address, 10.0.0.168.
- **Memory.** The default ceiling was 7 GB, too small for Parakeet plus Qwen3-TTS
  plus llama-server. Raised to 12 GB on a 16 GB host.

Delete that file and run `wsl --shutdown` to revert. `scripts/check_env.sh` now
fails loudly on either condition rather than letting a table session discover it.

### C3. Stack installed without sudo

The distro had no `pip`, `gcc`, `cmake` or `nvcc`, and `sudo` requires a password
this session does not have. So no apt and no source build of llama.cpp. Instead:
miniforge to `$HOME`, then the conda-forge `llama.cpp` **cuda130** prebuilt, and
`uv` to `$HOME/.local/bin` with speech-to-speech in its own venv. Paths are in
[versions.md](docs/versions.md). `check_env.sh` looks there before `PATH`.

### C4. A separate Linux venv for running the brain

The repo lives on the Windows filesystem, so its `.venv` holds Windows binaries and
cannot run in WSL. Running sets `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/brain`, which
keeps the Windows `.venv` usable for the Windows test run. The suite passes in both:
58 on Windows (Python 3.11.6), 58 in WSL2 (Python 3.12.3).

### C5. Windows paths inside TOML

`tests/test_app_integration.py` interpolates `tmp_path` into a TOML basic string.
A raw Windows path makes `\U` and `\s` escapes that `tomllib` rejects. The fixture
uses `.as_posix()`, which parses on both platforms.

### C6. `.gitattributes` forces LF on `*.sh`

The scripts run in WSL; a CRLF checkout breaks the shebang.

### C7. Real emotion names in `persona.example.yaml`

The plan's placeholders (`happy`, `surprised`, `thinking`, `proud`) are not in the
robot's library. The live dataset
`pollen-robotics/reachy-mini-emotions-library` on robot firmware 1.10.0 supplies
`laughing1`, `surprised1`, `thoughtful1`, `proud1`. Plan 3 should also note
`attentive1`, `attentive2`, `curious1`, `waiting` for the attend routine and
`sleep1` for the mute cue.

### C8. `check_env.sh` scope

Kept the plan's checks and added networking mode, memory ceiling, CUDA backend
presence and robot reachability, since each is a failure mode actually hit here.

## D. Design-document open items now answered

From section 12 of the design spec, closed by inspecting the live robot and the
installed packages rather than by guessing:

| Open item | Answer | Evidence |
|---|---|---|
| Does the SDK expose sound direction from the mic array? | Yes | `GET /api/state/doa` on the robot daemon |
| How is `head_tracking` toggled? | Three daemon endpoints | `/api/media/tracking/enable`, `/disable`, `/face` |
| Does Gemma 4 E4B GGUF have a multimodal projector? | Yes | `mmproj-gemma-4-E4B-it-{BF16,Q8_0}.gguf` in `ggml-org/gemma-4-E4B-it-GGUF`, so Plan 3 needs no SmolVLM2 |
| Which emotion names exist? | 84 moves enumerated | `/api/move/recorded-move-datasets/list/...` |
| Does speech-to-speech do the tool round trip? | Yes, for robot-side tools | `base_openai_compatible_language_model.py:543` |
| Which Responses events are consumed? | Exactly three | `responses_api_language_model.py:178-193` |

Also newly available and worth considering in a later plan: `/api/media/wobbling/{enable,disable}`
for idle liveliness, and speech-to-speech's `--smart_turn` turn detector, which may
do better than a fixed VAD threshold at a noisy table.
