# Deviations from Plan 1

Every departure from `plans/2026-09-10-reachy-familiar-plan-1-core.md`, with the reason
and where to check it. Plan 1's own test total was 49; the suite now runs 64, the extra
15 being the tests that cover the defects below. Sections A to C came from reading the
plan and the installed packages; section E came from running the real stack.

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
Windows (Python 3.11.6) and WSL2 (Python 3.12.3).

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

## E. Found by running the real stack, not by reading it

These only surfaced once Gemma, Parakeet and Qwen3-TTS were actually running. Each
change below is backed by a measurement recorded in [versions.md](docs/versions.md).

### E1. `--jinja` is required or no tool can ever fire

Without it llama-server ignores the `tools` array entirely and returns
`tool_calls: null` with empty content, despite the model's own chat template
declaring tool support. Verified both ways against the live server. Added to
`scripts/start_all.sh` with a comment, because a future tidy-up that drops it would
silently disable `roll`, `initiative` and `note` rather than fail loudly.

Related, and worth knowing: llama.cpp's `/v1/responses` accepts only the **flat**
Responses-style tool schema. The nested Chat-Completions shape returns HTTP 500,
`Failed to parse tools: key 'name' not found`. The brain already sent the flat form,
so no change was needed, but the error message is unhelpful enough to record.

### E2. Gemma 4 is a reasoning model, which broke both latency and tool use

Left alone it emits a `reasoning` output item before anything else, spending the
output budget thinking and then answering in prose. Two consequences, both fixed by
defaulting `reasoning: {"effort": "none"}` on upstream requests:

| | before | after |
|---|---|---|
| addressed turn, first token | 9278 ms | 183 ms |
| addressed turn, total | 9780 ms | 620 ms |
| follow-up turn, total | 8557 ms | 505 ms |

The design document's latency ceiling is 2.5 s with a 1.5 s target. Before this the
brain missed the ceiling by roughly 4x; after it, every addressed path including
tool round trips lands under a second. A client that sets `reasoning` itself, via
speech-to-speech's `--responses_api_reasoning_effort`, still wins.

### E3. The familiar invented dice results 5 times in 12

With the persona as the plan shipped it, the model called the `roll` tool in only
**7 of 12** live trials and made a number up in the other five. Design 4.5 says the
tool does all arithmetic and the model reports its total verbatim, so this is a
correctness failure, not a style one, and a bad one for a table that has to trust
the thing.

Adding a tool-use line to the **non-overridable** fixed blocks in `persona.py` took
it to **12 of 12**. It sits in the fixed blocks rather than `persona.yaml` precisely
so a future persona cannot delete it. Temperature 0 was also tried and added nothing
over the instruction.

Confirmed end to end afterwards: four consecutive dice turns, every spoken number
matching the tool's own total, advantage correctly keeping the higher of two dice.

### E4. Nothing recorded what the familiar actually rolled

The tool registry executed tools and returned the result to the model, but logged
nothing, so there was no way to audit a disputed roll after the fact. It now logs
every call and result at INFO. This also exposed that my first attempt at verifying
dice execution, grepping the brain log, was checking for something the code never
wrote. The registry now writes it.

### E5. The gate classifier was timing out on every turn

`classify_timeout_s` is 0.6 and the classifier call was taking about 286 ms warm but
exceeding the budget whenever slot two was cold or contended, which silently forced
every layer 3 decision to IGNORE. That is the designed fail-safe, so nothing looked
broken, but AMBIENT could never fire and an addressed line that omitted the name was
always missed. Adding `reasoning_effort: none` to the classify body took it to 59 ms,
about a tenfold margin. The timeout value itself is left alone for the Plan 4 tuning
pass.

### E6. The follow-up window never closed, so the familiar answered everything

This is the most serious thing the live run turned up, and it is a flaw in the design
rather than in the plan's transcription of it.

Design section 5, layer 2, opens the gate when "the familiar finished speaking fewer
than N seconds ago". Speaking is what re-arms that window. At a table where somebody
says something every few seconds, the window therefore never closes: the familiar
speaks, which re-arms it, so the next utterance is ADDRESSED, so it speaks again.

Observed live, six consecutive turns, none of them addressed to it:

```
{"gate_ms": 0.0, "generation_ms": 359.4, "decision": "ADDRESSED", "spoke": true}
{"gate_ms": 0.0, "generation_ms": 583.1, "decision": "ADDRESSED", "spoke": true}
{"gate_ms": 0.0, "generation_ms": 370.6, "decision": "ADDRESSED", "spoke": true}
... and so on
```

`gate_ms: 0.0` is the tell: no classifier call, the window alone was granting every
turn. This fails the design's own headline success criterion, "speaks zero times when
not addressed", and it would have done so for a whole four-hour session.

The plan's integration test could not catch it, because it simulates the window
closing by reaching into state and subtracting sixty seconds. Nothing in the code ever
closed it.

Fix: cap the number of consecutive follow-up grants, `max_follow_ups` in
`[gate]`, default 3. Naming the familiar, or pressing Speak Now, resets the count and
starts a fresh conversation. After the cap the gate hands back to the classifier.
Covered by `test_follow_up_window_does_not_chain_forever` and
`test_explicit_address_resets_the_follow_up_chain`.

**This one is worth TJ's judgement.** A cap of 3 preserves a natural short exchange
while guaranteeing the chain ends. If a different shape is wanted, for example
anchoring the window to the last time the familiar was named rather than to the last
time it spoke, that is a design decision rather than a bug fix.

### E7. The startup warmup probe opened the window before anyone spoke

Related, and the thing that exposed E6. The client's `warmup()` sends "Hello" at
pipeline startup. The classifier called that ADDRESSED, the brain answered it, and the
non-streaming path called `mark_spoke`, so the follow-up window was already open before
a single word had been said at the table. The first real utterance was then treated as
a follow-up.

Fix: the non-streaming path no longer calls `mark_spoke`. It serves the warmup probe
and history compaction, neither of which is speech in the room. Covered by
`test_warmup_does_not_open_the_follow_up_window`.

### E8. The layer 3 classifier read Dungeon Master narration as something to react to

Once E5 made the classifier actually respond instead of timing out, its judgement
became visible and it was poor. Scored against twelve labelled lines of table talk:

| Prompt | score | notes |
|---|---|---|
| the plan's wording | 7/12 | called narration such as "You see a rusted iron door" AMBIENT |
| revised wording | 10/12 | both misses fall toward silence |

The revised instruction states plainly that almost everything at the table is between
the humans, lists Dungeon Master narration and rules questions as IGNORE explicitly,
and ends with "when in doubt answer IGNORE". Both of its remaining misses are
conservative: it stays silent on "Roll me a d20 would you?" and on a burst of alarm it
could have reacted to. Given the design asks for zero uninvited speech, erring that way
is the right trade. The two misses are worth revisiting in the Plan 4 tuning pass.
