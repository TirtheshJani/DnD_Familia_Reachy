# Deviations from Plan 1

Every departure from `plans/2026-09-10-reachy-familiar-plan-1-core.md`, with the reason
and where to check it. Plan 1's own test total was 49; the suite now runs 68, the extra
19 being the tests that cover the defects below. Sections A to C came from reading the
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

### E9. The classifier's context window made it worse, not better

Design section 5 says layer 3 sends "the last four segments". Once E8's prompt was in
place the familiar still spoke uninvited, and the cause was the context itself: as soon
as any earlier segment named the familiar, the model read every line after it as part
of one continuing conversation. Live example, verdict ADDRESSED on plain narration:

```
- Understudy, roll me a d20 with advantage.
- Hello
- the door creaks open somewhere in the dark
```

Scored on twelve labelled lines, several of them carrying exactly that contamination:

| Classifier input | score | would speak uninvited |
|---|---|---|
| last four segments, as the design specifies | 4/12 | 7 times |
| the last line alone | 11/12 | **0 times** |

An explicit "the earlier lines are context only" caveat helped but did not close it:
7/10 with 3 uninvited. Dropping the context entirely did.

This is safe because layers 1 and 2 already handle the two cases the context was there
to serve: layer 1 catches the familiar being named, and layer 2 catches the moment just
after it spoke. Layer 3 only ever sees the residue, and for that the last line is the
whole question. `Gate` still hands the classifier the recent segments, so reinstating
context later is a one-line change.

The single remaining miss is conservative: "He went down! Get him up, get him up!"
reads as IGNORE rather than AMBIENT, so the familiar stays still instead of reacting.
Worth revisiting in the Plan 4 tuning pass, along with E8's two misses.

---

# Deviations from Plan 2

`plans/2026-09-11-reachy-familiar-plan-2-live-loop-and-knowledge.md` and its
`revision-b` companion. Same rule as above: the behaviour the plan asks for, tested
first; the structure of the code as it actually stands.

## P1. The follow-up window anchors to naming (Task 1)

Replaces design section 5 layer 2, and supersedes the cap-only fix recorded in E6.
The window now opens when the familiar is named, aliased, or armed by Speak Now, and
its deadline is fixed at that moment plus `follow_up_window_s`. The first reply after
naming may push the deadline out once; later replies do not. `max_follow_ups` survives
as a second guard for a rapid exchange inside a single open window.

E6 capped the chain but left it anchored to speech, so a burst of three uninvited
replies was still reachable after any single naming. Anchoring removes the re-arm
entirely: after the window closes, only naming reopens it.

Covered by `tests/test_gate_anchoring.py`, four tests.

### P1a. The counter keeps its Plan 1 name

The plan's snippet calls it `state.follow_up_grants`. The attribute already existed
as `state.follow_ups` with `reset_follow_ups()` and `count_follow_up()`, so those were
kept and `mark_named` calls `reset_follow_ups()`. Behaviour is what the plan specifies;
only the spelling differs.

### P1b. The plan's own anchoring test did not exercise the extension

`test_reply_to_reply_does_not_re_arm_window` as written calls bare `st.mark_spoke()`,
which stamps `time.time()`. No fake clock can reach that value, so
`last_named_at <= spoke <= deadline` was always false and the single-extension branch
was never entered. The four assertions still passed, for the wrong reason. Every
`mark_spoke` in the anchoring tests now passes an explicit `at=`.

### P1c. Five Plan 1 tests needed the clock advanced

Predicted by the plan and confirmed: `test_alias_word_boundary_only`,
`test_speak_now_arms_one_turn`, `test_follow_up_window_open_then_closed`,
`test_classifier_used_only_when_earlier_layers_say_no`, and
`test_mock_session_speaks_only_when_addressed`. Each named the familiar and then
expected IGNORE on the next line with the clock frozen. Fixed by advancing a mutable
clock past the window, or in the integration test by backdating `last_named_at`
alongside `last_spoke_at`. No assertion was weakened. The stale docstring on
`test_follow_up_window_does_not_chain_forever`, which described the speech anchoring
this change removes, was rewritten to describe what the test now pins.

## P2. Revision B's `nephria-context/` is this repo's `plans/0*.md` (Task R1)

R1 copies the curated set from `nephria-context/`. No such directory exists. The set is
`plans/00-README.md` through `plans/09-lexicon.md`, which `plans/00-README.md` itself
confirms: "every file here can be dropped into `data/notes/` and indexed as-is". Those
ten files were copied, not moved, and `05-house-rules.md` became
`data/notes/homebrew/house-rules.md`.

R1 also moves "the seven recorder .txt files" into `data/raw/`. No such files exist
anywhere in the repo, so `data/raw/` was created empty. Nothing was lost; there was
nothing to move.

### P2a. `data/notes/` is not tracked by git

`.gitignore` has excluded `data/` since Plan 1, so the curated corpus is local only and
a fresh clone starts with no notes. The tracked source of truth stays `plans/0*.md`.
`docs/notes-policy.md` gained a "Seeding a fresh clone" section with the three commands
that rebuild the working copy, and a warning that anything written into `data/notes/`
afterwards, session recaps most of all, exists on one machine until somebody writes it
back into `plans/`.

## P3. Task ordering: R2 and R4 partly land inside base Task 10

Revision B slots R2 before base Task 9 and R4 before base Task 10, but R2 Step 4 edits
`knowledge.CORPORA` and R4 Step 5 adds an assertion to `tests/test_app_knowledge.py`,
and base Task 10 is what creates both of those files. Writing them twice would have been
theatre, so:

- R2's `index_notes` change ran in its own commit before Task 9, as the revision asks.
- R2's `CORPORA = ("homebrew", "rules", "notes", "sessions")` was written into
  `knowledge.py` when Task 10 created it, together with R2's ordering test.
- R4's `lexicon.py`, `config/lexicon.tsv` and config wiring ran before Task 10; R4's
  `app.py` wiring and its integration assertion landed in Task 10's commit, because both
  tasks edit the same handler and editing it twice invites a merge error for no gain.

Likewise R3 replaces base Task 9's `lookup_rule.run` outright, so only the R3 version was
ever written and the two tasks share one commit.

## P4. `[paths] players` instead of a hardcoded path (Task R5)

R5 has `create_app` pass `load_players("config/players.txt")` literally. That resolves
against the working directory and would let a real `config/players.txt` change what the
test suite does. It is now `cfg.paths.players`, defaulted to `"config/players.txt"`, so
the default behaviour is identical and the integration fixture can point somewhere else.
`paths.lexicon` is defaulted the same way and for the same reason.

## P5. `pymupdf` rather than `fitz`

Both plans' snippets `import fitz`. The installed PyMuPDF emits a DeprecationWarning for
that name and asks for `import pymupdf`. Used in `familiar_brain/chunking.py` and
`tests/test_chunking.py`. No behavioural difference.

## P6. Corpus scoping had to move inside the KNN query (Task 6)

The plan filters by corpus in SQL *after* taking the `k * 8` nearest neighbours, and
notes that the over-fetch is there "to keep recall when corpora are mixed". It does not.
Demonstrated with a test before changing anything: with 2000 rules chunks indexed beside
one notes chunk, all sharing vocabulary with the query, every candidate came back a rules
chunk and `search(corpus="notes")` returned **zero** hits. In production that is the
normal shape of the store, two SRDs against a handful of notes, so `recall` and the
notes and sessions lines of the "What you know" block would have been silently empty.

`corpus` is now a `vec0` metadata column, verified supported on sqlite-vec 0.1.9, so the
KNN search itself is scoped. Over-fetch is still needed, because `weight` can lift a
slightly more distant chunk, but it is now `k * 4` *within the corpus* rather than a
global candidate set. Covered by
`test_search_still_finds_a_small_corpus_beside_a_large_one`.

A store built before this change has the old `vec` schema and will not pick up the new
column, since `CREATE VIRTUAL TABLE IF NOT EXISTS` does not migrate. Delete
`data/familiar.db` and reindex. No such store existed when this landed.

## P7. Tests added beyond the plan's list

Each covers a public function or endpoint the plan created but did not test, which the
repo's review standard requires:

- `test_label_session_enter_keeps_the_gate_decision`, and
  `test_gate_report_with_no_spoken_turns_reports_none_not_a_crash` (empty latency list).
- `test_missing_file_is_an_empty_lexicon` and `test_canonical_spelling_is_left_alone`.
- `test_homebrew_files_are_skipped_on_a_second_pass_too`: `sources()` is per corpus, so
  without merging both corpora every reindex would re-chunk the house rules.
- `test_build_context_names_the_edition_of_a_rules_hit`.
- `test_summarise_twice_does_not_duplicate_chunks` and
  `test_purge_of_an_unknown_session_is_false_not_an_error`.
- `test_session_end_summarises_in_the_background`, `test_sessions_listing_then_purge`,
  `test_purge_of_an_unknown_session_is_false`, `test_reindex_endpoint_reports_counts`,
  `test_retrieval_is_timed_and_the_tools_are_offered`: the five endpoints Plan 2 adds.
- `test_lexicon_corrects_the_logged_and_forwarded_text`: R4's integration assertion,
  widened to check the session log as well as the forwarded request.

## P8. Lexicon entries that can fire on ordinary words

Implemented exactly as R4 specifies, but three of its variants are common English:
`sticks` becomes Styx, `Gem` becomes Jam, `Molar` becomes Alar. "We gathered sticks for
the fire" will be corrected to "Styx". The fuzzy pass is guarded (six characters
minimum, 0.86 similarity, canonical spellings left alone) but the exact-variant pass is
not, by design, because those are the mishearings the table actually produced. Worth
watching at the live loop: the Vocabulary line in `docs/live-loop.md` is what settles it,
and dropping a variant is a one-line edit to `config/lexicon.tsv`.

## P9. The retrieval seam, and what was not run

`indexing.open_store(cfg)` is the single seam the app and the CLI share, and the one
`tests/test_app_knowledge.py` monkeypatches to a `FakeEmbedder`-backed store. No test
downloads a model.

Three steps need a human in the room or files that cannot be redistributed, and were
deliberately not run:

- **Task 4, the live loop.** Needs the robot, the room, and TJ reading
  `docs/live-loop.md` aloud. `tests/fixtures/live_session_1.jsonl` does not exist yet.
- **Task 8 Step 5, indexing the SRDs.** The PDFs are not in the repo and are not
  redistributed. `README.md` carries the CC-BY-4.0 attribution and the two index
  commands. The `rules` corpus is empty until they are run, so `lookup_rule` currently
  answers from `homebrew` only.
- **Task 11 Step 6, summarising the real transcript**, and **Task 12 Step 4, the live
  check.** Both depend on Task 4.

What was run instead: the notes corpus was indexed for real with `BgeEmbedder` and
spot-checked across both corpora, recorded in [versions.md](docs/versions.md) under
"Notes index, first build". The `slow` embedder test was run once and passes.

## P10. `gate-report` survives a missing brain log (Task 2)

Found by running the CLI against a real session file rather than a fixture.
`docs/live-loop.md` has you type the brain log path by hand, and it lands after
labelling is already done, so a typo or a wrong platform threw
`FileNotFoundError: data\brain.log` and lost the whole report. On this Windows checkout
the file does not exist at all, because the service that writes it runs in WSL.

`gate_report` now reports the counts regardless and adds `"brain_log_missing": true`,
so a null latency cannot be read as "latency was fine". Covered by
`test_gate_report_says_so_when_the_brain_log_is_missing` and
`test_gate_report_omits_the_missing_flag_when_the_log_is_there`.

## P11. The embedder loads in 11.9 seconds, and it was loading on the first turn

Found by running the suite one more time before merging, after the model had been
downloaded. `tests/test_app_integration.py` started failing on its follow-up row with
`assert 'idle_do_nothing' not in ['idle_do_nothing']` and a
`retrieval failed: [Errno 2] No such file or directory` in the captured log.

The cause was not the test. That fixture never monkeypatched `indexing.open_store`, so
Task 10's wiring gave it a real `BgeEmbedder`, and `build_context` was loading the model
inside the turn. Measured on this machine:

| | first embed | warm embed |
|---|---|---|
| `BgeEmbedder` | 11863 ms | 21 ms |

Twelve seconds inside an addressed turn is two things at once. It is far past the design
section 11 ceiling of 2.5 s. More quietly, it is longer than the whole 8 s follow-up
window, so the window opened by the naming turn had already closed by the time the
player's follow-up arrived, the classifier judged it cold, and the familiar ignored it.
That is the exact failure the test caught, on the first live turn of every session.

Two fixes:

- `indexing.warm_store(store)` forces the load, and `open_store` calls it, so the cost
  is paid during `create_app`, behind the `/api/state` health check that
  `scripts/start_all.sh` already waits on for up to 600 s. The health check now goes
  green only when the brain can actually retrieve, which is the better contract anyway.
  A warm-up failure is logged and swallowed: a brain that will not start is worse than
  one with no memory. Covered by `test_warm_store_loads_the_embedder_up_front`,
  `test_warm_store_survives_an_embedder_that_cannot_load`, and the slow
  `test_open_store_returns_an_already_loaded_embedder`.
- The Plan 1 integration fixture now monkeypatches `open_store` to a `FakeEmbedder`
  store, as `tests/test_app_knowledge.py` already did. No test loads a 130 MB model or
  touches the network. The default suite went from 26 s to 16 s as a side effect.

Worth saying plainly: the earlier green suite was partly hollow. Retrieval was failing
on every turn of that test and the `except Exception` in `_instructions` was swallowing
it exactly as designed, so nothing looked wrong. The graceful degradation is right, but
it means a silent retrieval failure is invisible from the test suite alone. The
`retrieval_ms` mark in the brain log is what exposes it live, and
`test_retrieval_is_timed_and_the_tools_are_offered` pins that the mark exists.

## P12. File 10, the DM's canon, reaches the brain (2026-10-02)

`plans/10-nesh-canon.md` arrived with a "Canon corrections" section in `00-README.md`
and twelve new rows in `09-lexicon.md`. None of it was in `data/notes/`, and the seed
recipe in `docs/notes-policy.md`, `cp plans/0*.md`, would never have copied file 10.
The recipe now copies `plans/[0-9][0-9]-*.md`, which takes 00 to 10 and leaves out the
dated plan documents. Running it in a scratch directory reproduces today's `data/notes/`
exactly, line endings aside.

Retrieval tags a chunk with its corpus and heading, never its file, so a chunk from
`03-nephria-city.md` saying "North Reach Plaza" would reach the model with the same
authority as file 10. The `data/notes/` copies of 03 (two lines) and 09 (one row) now say
Northreach Crossing; `plans/` is untouched, and the recipe repeats the rename (with
`perl -pi -e` since P16).
No other superseded spelling was found: Thanatos appears only as a lexicon variant.

`config/lexicon.tsv` gained the canon's likeliest mishearings: the old settlement name
and its split forms, Thanatos, the Gatewarden, Maliburn and sennight split in two. Eris
and Nesh get no rows, being under the six-letter fuzzy floor with everyday neighbours
(iris, error, mesh, fresh). Neshians and sennights have bare canonical rows only so the
fuzzy pass cannot strip their plural. Covered by
`test_shipped_lexicon_corrects_heard_forms_of_the_nesh_canon`,
`test_shipped_lexicon_leaves_ordinary_words_near_the_nesh_canon_alone` and
`test_shipped_lexicon_keeps_the_plurals_of_the_nesh_canon`.

Reindexed with the real `BgeEmbedder` (4 files indexed, 7 unchanged). File 10 is the top
notes hit for "What is the Song?", "Who is Thatos?", "What are the Twelve Declarations?"
and "where is Northreach Crossing". The old name now reaches the prompt only inside
sentences that correct it. That check covered four questions only, and counted ranks, not
what reached the model; P16 found file 10 cut short at the prompt and still missing for
"what is a sennight", where it is not in the top three.

## P13. Rocky replies only when spoken to (2026-10-02)

Live on 2026-09-16 (`data/sessions/20260916-182232-0189cc.jsonl`), the one time in about
seventy lines that Rocky spoke uninvited was an 868 word segment: the transcriber had
merged minutes of table chatter, and in the middle of it players explained the familiar
to each other ("we're doing uh rocky", "What's his name, sorry? Rocky."). `_named()`
found the name anywhere in the segment, so the gate said ADDRESSED and Rocky generated
for a minute. Three changes, all in `familiar_brain/gate.py` and `llm.py`:

- **A naming must be spoken to Rocky, not about it.** In a run-on segment (over 60
  words, since P16) it counts only when at most 25 words follow it, since the line just
  said to the familiar is the last thing the transcriber glued on; words rather than sentences because transcripts often
  come back unpunctuated. The live segment had 64 words after its last "rocky". A small
  set of mention shapes, in the style of the adjective rules, also stop counting: the
  name introduced as a name ("his name is", "called", "named", "I call him"), "we're
  doing rocky", "talking about Rocky", the name alone answering "What's his name?", and
  a possessive ("Rocky's eyes are glowing"), except "Rocky's turn" and "Rocky's go".
  This does not stop every line about Rocky: a short third person statement ("Rocky is
  so cute", "I think Rocky is broken", "look at Rocky") still reads as a naming, since
  nothing in its words tells it from one said to Rocky. That is the limit of gating on
  the name, and `test_a_short_statement_about_rocky_still_reads_as_a_naming` pins it.
- **The classifier can no longer make Rocky speak.** `CLASSIFY_INSTRUCTION` offered
  ADDRESSED for "a direct question or command that only Rocky could be expected to
  answer", which supersedes part of E8. It now chooses only AMBIENT or IGNORE, keeping
  its lean to IGNORE, and `LLMClient.classify` no longer parses ADDRESSED. If any
  classifier still answers ADDRESSED, the gate logs it and returns IGNORE. Speech now
  needs a naming, the follow-up window after one, or Speak Now.
- **A run-on inside the follow-up window is chatter, not a follow-up.** Over 60 words
  (twenty seconds and more of unbroken speech; real follow-ups have run to about ten)
  goes to the classifier instead, and does not use up one of `max_follow_ups`.

The live row is kept verbatim in `tests/fixtures/live_uninvited_rocky.jsonl`, since
`data/` is not tracked. Covered by `tests/test_gate_spoken_to.py` and
`test_classify_only_tells_ambient_from_ignore`. The new instruction has not been scored
against the live model; its only job now is AMBIENT against IGNORE, and a miss either way
can cost a reaction but never a reply.

## P14. Head tracking is always on while Rocky is awake; Plan 3 D4 reversed (2026-10-02)

Plan 3 D4 had the face tracker on only while Rocky was addressed and for
`tracking_linger_s` after the follow-up window, and off otherwise so that a `move_head`
toward the daemon's sound direction could show on the other lines. TJ asked for the
face tracking to be always on. It now is, on every turn that carries a preface (IGNORE,
AMBIENT, ADDRESSED, the greeting and sight continuations), from the first line after
the brain starts, whoever that line was said to. Muted is asleep: tracking goes off on
the next turn, ahead of the sleepy cue, and comes back on at the turn after an unmute,
ahead of the welcoming cue. Toggles still go out only on a change, since a repeated
`head_tracking(true)` resets the app's speaking handoff.

`move_head` is invisible while the tracker owns the head at weight 1, so the glances
could never be seen again and went rather than stay as dead code: the glance path in
`attention.py`, `glance_item`, `robot.py` and its DoA reader (nothing else read the
daemon), the glance stamp in `BrainState` and in the revision rewind's `_PrefaceStart`,
and the `[robot]` section and `glance_gap_s` and `tracking_linger_s` keys in
`config/brain.toml`. The config loader still has to accept a local `brain.toml` that
has them, and unknown keys raise there, so it skips the `[robot]` section and drops the
two retired motion keys by name; any other unknown key still fails.

Covered by `tests/test_attention.py`,
`test_the_first_line_switches_tracking_on_even_when_nobody_spoke_to_rocky`,
`test_tracking_stays_on_across_every_kind_of_turn_with_no_repeated_toggle`,
`test_unmute_switches_tracking_back_on_ahead_of_the_wake_cue`,
`test_greeting_prompt_switches_tracking_on_ahead_of_the_wake_gesture_and_opens_no_window`,
`test_a_revised_first_line_nobody_spoke_to_rocky_still_switches_tracking_on` and
`test_a_brain_toml_with_the_retired_glance_keys_still_loads`. Smoke-test P2 now checks
that tracking stays on and sleeps with the mute, instead of the hand-back and glance.

## P15. More expression from the keyword rules, none over plain replies (2026-10-02)

TJ asked for Rocky to be more expressive. Live on 2026-09-16 (`data/brain.log`) the
reaction call answered none on 49 table lines, timed out at 1.5 s on 4 and chose a
reaction once, so nearly all of Rocky's expression between replies has to come from the
free keyword rules in `familiar_brain/motion.py`. Commit 164b7f7 stands: a plain reply
still plays no emotion.

- **More table rules.** Shock ("oh no", "oh my god", "what the hell", "are you
  kidding", "holy crap", and "no way" only as an exclamation of its own, so "no way to
  get across" is not a shock) plays surprised. A victory ("Let's go!" or "let's gooo"
  but not "let's go to the tavern" or "Okay, let's go.", a shouted "Yes!", "we did it"
  and "we won" where they end the sentence, "nailed it", "huzzah", "victory", "woohoo")
  plays the new `cheer` mood, success. (P16 narrows several of these.) "Killed it" is left out: at this table it is as
  likely the wolf that killed the horse. "Roll initiative" leans in with attentive;
  no reply follows a table line, so it plays over no speech. "Rest in peace", "that's so
  sad" and "she didn't make it" join sad, and a transcriber's "[laughter]" joins
  amused. The specific rules keep first place, so "oh no, he went down" stays scared.
- **`cheer` is a reaction mood.** It is in `REACTION_MOODS` and `REACTION_INSTRUCTION`,
  split out of amused ("funny, or a triumph for the party") so a victory narrated at
  length, which names no keyword, gets success rather than happy. Not yet scored
  against the live model. Both persona files map it to `play_emotion:success`.
- **An AMBIENT line always reacts.** The gate answers AMBIENT only for a burst of
  excitement or dismay, which is short by nature, so `for_table(..., ambient=True)`
  asks the reaction call whatever the word count, and when the call fails it plays
  surprised (P16: an explicit none no longer does) (`AMBIENT_FALLBACK_MOOD`): engaged for a cheer and a groan
  alike, where amused would grin at a friend going down. A table rule mood that a
  persona has no gesture for is now left to the call, as on an addressed turn.
- **A greeting or a goodbye waves.** An addressed line that is only "hello", "good
  evening", "welcome back", "goodnight", "bye" or "see you next session", with a word
  of address and for a greeting a "how are you", plays welcoming. "Hey" is not a
  greeting: it comes before nearly every request, and a wave on each would bring back
  the constant emoting. A goodbye waves rather than plays sleepy, which is the mute
  cue's gesture and would read as Rocky going to sleep.
- **`reaction_gap_s` 15 to 10**, in `config/brain.toml` and the `Motion` default, since
  most reactions now cost no call. `min_gesture_gap_s` stays 6.
- **The manner is a touch warmer in words**: "a quick spark of delight or dismay when
  the dice turn" and permission to open with a short exclamation such as "Oh!" or "Ha!". No
  stage directions, which made Gemma speak robot tool calls (commit 9e8fe86).

Tests that used "Rocky, hello" as the plain reply now use "Rocky, and the docks?". New:
`test_table_rules_catch_shock_at_the_table`,
`test_table_shock_rules_leave_ordinary_table_talk_alone`,
`test_table_rules_cheer_a_victory`, `test_table_victory_rules_leave_plans_and_answers_alone`,
`test_table_rules_lean_in_when_combat_starts_and_mourn_a_little_more`,
`test_table_rules_keep_their_order_when_a_line_says_two_things`,
`test_addressed_rules_greet_and_see_off_but_only_a_line_that_is_all_greeting`,
`test_a_greeting_plays_welcoming_and_a_plain_reply_still_plays_nothing`,
`test_a_victory_at_the_table_plays_success`,
`test_an_ambient_burst_asks_the_reaction_call_however_short`,
`test_an_ambient_burst_falls_back_to_surprised_when_the_call_fails_or_says_none`,
`test_an_ambient_burst_respects_both_gaps`,
`test_an_ambient_burst_with_no_rule_and_no_reaction_still_looks_surprised`,
`test_an_ambient_burst_plays_what_the_reaction_call_chose`,
`test_a_victory_shouted_at_the_table_plays_success_without_a_reaction_call`,
`test_a_greeting_to_rocky_waves_but_a_request_after_hey_does_not` and
`test_reaction_prompt_tells_a_victory_from_a_joke`.

## P16. Review fixes to P12 to P15 (2026-10-02)

A review of the day's work found these, each reproduced before it was fixed.

- **Recall questions to Rocky were read as mentions** (`gate.py`). "what was the
  innkeeper's name? Rocky?" matched the name-question rule, and with the comma dropped
  "what's this called Rocky" or "what were we talking about Rocky" matched the mention
  words. The name-question rule now needs his, her, its or their name (or "call him?")
  and a bare answer, the name then a stop or nothing. The mention words do not count in
  a clause that opens with a question word when the name ends it or comes before its
  "?". "we named him Rocky" and "What do you call him? Rocky." are now mentions too.
- **The 25 word rule applies only to run-ons** (over 60 words, `_MAX_FOLLOW_UP_WORDS`).
  A 34 word recap request that starts with "Rocky" was being dropped.
- **A new app session gets tracking back** (`app.py`). The greeting opens one, whose
  MovementManager starts with tracking off, but the brain still held it on and toggles go
  out only on a change, so the head stayed still all night. The greeting now resets the
  brain's flag first.
- **An AMBIENT line answered none plays nothing** (`motion.py`). The fallback overrode an
  explicit none, so narration the classifier calls AMBIENT ("You descend into the
  cistern.") always got surprised. It now plays only when the call fails, gives no usable
  answer, or picks a mood the persona has no gesture for.
- **Table rules read the last 25 words of a segment**, and "oh no", "no way", "holy
  crap", "are you kidding" and "what the hell" must end their sentence. Of the six live
  "oh no" hits in `data/sessions`, the three in 70 to 92 word chatter go; the three where
  "Oh no." ends its sentence ("Oh no. I'm sorry.") still play surprised. "Victory",
  "hooray", "hurrah" and "huzzah" count only as a cry of their own, not in "Victory Road"
  or "roar in victory". "Roll initiative" does not count inside a question and no longer
  spends the reaction gap, so the first crit of the fight still reacts.
- **Unpunctuated triumphs**: "YES YES YES", "Fuck yes", "we killed the dragon!" and "we
  got him!" cheer, and "rolled a twenty" joins the crit rule. "No no no" is a shock. Not
  "crits", since "he crits you" is the monster's.
- **The lexicon keeps possessives** (`lexicon.py`). The fuzzy pass read "Gatewarden's"
  as one token and rewrote it to "Gatewarden", and "Neshian's" to "Neshians". A trailing
  possessive is now matched without its apostrophe and given back, and a plain plural of
  a canonical word is left alone.
- **Notes are chunked at 700 characters**, the cut `build_context` makes, not 1200
  (`chunking.py`), and a list number stays with its item. File 10's Twelve Declarations
  was one chunk cut inside declaration 8. Reindexed (11 files): "What does the Hospitality
  declaration say?" now brings the Hospitality text into the prompt.
- **File 04's "Styxian gods such as Eris"** is corrected in the `data/notes/` copy and
  the seed recipe, which now uses `perl -pi -e` because BSD and macOS `sed -i` reads the
  script as a backup suffix. After reindexing, "Who is Eris?" no longer puts the stale
  sentence in the prompt. Still open: "what is a sennight" has no file 10 chunk in its top
  three, and "who is the Gatewarden" ranks file 10 third.
- **Docs**: smoke-test P3 reads a and b again five seconds apart, not ten, since
  `reaction_gap_s` is now 10. The `CLASSIFY_INSTRUCTION` comment says its 10 of 12 score
  was the three way wording's; the AMBIENT or IGNORE wording is unscored.

New tests: `test_a_naming_buried_early_in_a_run_on_segment_does_not_count` (renamed, now
over 60 words), `test_a_long_request_that_starts_with_the_name_still_counts`,
`test_a_short_statement_about_rocky_still_reads_as_a_naming`, new cases in
`test_talking_about_rocky_is_not_a_naming` and `test_speaking_to_rocky_is_still_a_naming`,
`test_a_second_greeting_switches_tracking_on_again_for_the_new_app_session`,
`test_an_ambient_line_the_reaction_call_answers_none_plays_nothing`,
`test_an_ambient_line_the_reaction_call_answers_none_still_idles`,
`test_an_ambient_reaction_with_no_gesture_falls_back_to_surprised`,
`test_table_rules_read_only_the_end_of_a_long_segment`,
`test_shock_needs_an_exclamation_not_a_question_or_a_greeting`,
`test_victory_is_an_exclamation_not_a_name_or_narration`,
`test_unpunctuated_triumphs_still_cheer`,
`test_initiative_leans_in_without_costing_the_first_crit_its_reaction`,
`test_shipped_lexicon_keeps_possessives_and_plurals_of_the_canon`,
`test_a_misheard_possessive_is_corrected_and_keeps_its_suffix`,
`test_every_item_of_a_long_canon_list_reaches_the_prompt` and
`test_the_seed_recipe_corrects_eris_and_runs_on_macos_too`.
`test_an_ambient_burst_falls_back_to_surprised_when_the_call_fails` lost its "none" case.

## P17. Rocky learns during and between sessions without `/api/session/end` (2026-10-02)

The summariser ran only on `POST /api/session/end`. The one real session,
`data/sessions/20260916-182232-0189cc.jsonl`, was never ended, so nothing from it was
learnt, and `apply_retention` would have deleted its raw log unlearnt once it was a week
old. Mid-session, nothing said at the table was searchable at all.

- **Live indexing, embedding only** (`familiar_brain/live_index.py`). A worker reads the
  active session log every 5 s and indexes it into `sessions` once the oldest unindexed
  row is 30 s old (`REVISION_WINDOW_S`, so most revisions have landed) or 8 rows have
  piled up. It never calls the LLM: live on 2026-09-16 classify calls timed out at 1.5 s
  and a reply took 60 s. Only the changed tail piece is embedded again
  (`Store.delete_pieces`), so a batch costs a piece or two however long the session. It
  reads the file, not the turns, so lines dropped by mute-and-stop-logging never reach it.
  Chunks go under the raw log's own name, which the summariser, retention and purge
  already delete. Deliberate notes are indexed on the next tick at weight 2.0.
- **Revisions de-duplicated** (`familiar_brain/transcript.py`, shared with the
  summariser). The live log holds each utterance 2 to 6 times; a later line that repeats
  at least 60 percent of the previous line's words, in order, replaces it, keeping any
  front the transcriber's window slid past. The 2026-09-16 log goes from 70 rows and
  15607 characters to 38 and 10067. Deliberate notes are never folded.
- **Catch-up learning** (`familiar_brain/learning.py`). Every raw log with no summary,
  other than the active session, is queued at brain startup, `/api/session/start` and the
  new `POST /api/learn`, and summarised in the background one at a time, never while a
  session is active: one already running finishes, the rest wait for the next end, start
  or restart. A failure leaves the raw log for the next catch-up. A session with nothing
  logged gets a one-line summary and no model call. `familiar-brain learn` does it all
  now, synchronously, with progress; with llama-server down it reports that session as
  failed, keeps its raw log, and exits 1.
- **`/api/session/start` with a session open ends and queues it** instead of starting
  over it and orphaning it. The response gains `ended`.
- **Retention only deletes a summarised raw log**, and never one the learner holds.
  Catch-up is queued before retention runs at startup.
- **Deliberate notes are copied into the summary** under "Deliberate notes", by code, so
  they outlive the raw log; they are scrubbed like the rest. The summary prompt now gets
  the de-duplicated transcript.
- **A missing players file is a WARNING** at startup and before every summary, since
  `config/players.txt` is absent on the development machine and nothing is scrubbed
  without it. Nothing creates it. The file is now read afresh for each summary.
- **Purging the session in progress is refused** (409): the live indexer and the log would
  bring it back.
- `Store` takes a lock around its SQL, now that the live worker, the learner and request
  threads share one connection.

Not run against the real stack. On the next brain start the 14 development logs of
2026-09-11 (mock cistern sessions) are learnt too, before the 16 September one; purge
any not wanted with `DELETE /api/session/<id>` first. The 16 September transcript is about
4k tokens, well inside one slot, but a full four hour session will not fit one summary
call; that is not addressed here.

New tests: `tests/test_transcript.py`, `tests/test_live_index.py`, `tests/test_learning.py`
and `tests/test_app_learning.py`.

## P18. Long sessions are summarised in windows; a running campaign memory (2026-10-02)

P17 left two gaps. A four hour session does not fit one summary call: llama-server runs
`-np 2 -c 32768`, about 16k tokens a slot, and the 21 minutes logged on 2026-09-16 were
10067 characters de-duplicated, so four hours is about 115k (about 29k tokens). And
nothing joined the sessions up: "what should we do next, Rocky?" could only reach
whichever session summary it happened to match.

- **Map-reduce summaries** (`familiar_brain/summariser.py`). After de-duplication, lines
  made only of filler words ("Yeah.", "Uh.", "Okay okay.") are dropped; deliberate notes
  never are. A transcript of up to `WINDOW_CHARS` (24,000 characters, about 6k tokens at
  4 characters a token) still takes one call with the unchanged `SUMMARY_PROMPT`. A longer
  one is packed, whole lines in order, into windows of at most that size (a run-on line
  with no full stop is cut at a space), each summarised with `PART_PROMPT`, and the
  partials merged with `MERGE_PROMPT` into the same six headings, in stages if together
  they would not fit one window. Each call is then under about 7.4k tokens with prompt
  and reply, and under 9.5k even at 3 characters a token, well inside a slot. Deliberate
  notes go verbatim to the last merge under "## Deliberate notes", and are still appended
  by code as before. Every partial and the result are scrubbed of player names.
- **All or nothing.** An exception or an empty reply from any call leaves no summary
  file and no change to the store: the deletes of the session's old chunks now happen
  after the calls, so the live indexer's chunks stay searchable, and the file is written
  through a temporary name. The session stays unsummarised, so catch-up retries it.
- **Campaign memory** (`familiar_brain/campaign.py`). After each summary the model
  merges the previous memory and the new summary into `data/sessions/campaign-memory.md`
  (not `data/notes`, which people curate), with the headings Story so far, NPCs, Places,
  Open threads, Loot and resources, Decisions that matter. The prompt carries the same
  fiction-only rules, a 6,000 character cap (enforced by cutting at a line, with a
  WARNING, if the model ignores it), carries unresolved threads forward and drops
  resolved ones, and says it records only play: the DM's notes are canon and win. The
  notes themselves are not passed in. The version before each merge is kept as
  `campaign-memory.prev.md`. It is indexed into `sessions` under its own source at
  weight 2.0, one chunk per heading, and the Open threads and Story so far chunks carry a
  short cue in their title ("what the party could do next", "a recap") so a question that
  names neither still finds them. It is never added whole to a prompt.
- **Keeping it honest.** Its first line records each summary it was built from with a
  hash of the summary's text. If the summaries no longer match (one purged,
  re-summarised, or an older session learnt late) it is rebuilt from every remaining
  summary, oldest first, one merge at a time, rather than asking the model to forget one
  session. A purge drops the memory, its `.prev` too if that holds the session, and their
  chunks at once with no model call, since a purge can come mid-session; the learner
  rebuilds it at the next idle moment (`DELETE /api/session/<id>` kicks it). A merge
  failure does not fail the session's summary; the memory stays due and the next
  catch-up or `familiar-brain learn` retries it. No merge runs while a session is active.
- `GET /api/sessions` lists only session logs and summaries, not the campaign files.

New tests: `tests/test_campaign.py`, the map-reduce tests in `tests/test_summariser.py`,
and one in `tests/test_app_learning.py`. `test_nothing_is_summarised_while_a_session_is_active`
now counts two calls, the summary and its campaign merge.

## P19. Review fixes to P17 and P18: purges, pauses, and what reaches the prompt (2026-10-02)

A review of P17 and P18 reproduced each finding below in a scratch directory before it was
fixed. Each fix has a test.

- **A purge during a summary is no longer undone** (`familiar_brain/summariser.py`). The
  summary's model calls take minutes; a `DELETE /api/session/<id>` landing in them deleted
  the log and its chunks, then the summary wrote its file, re-added its chunks and the raw
  transcript's, and the campaign merge folded the session back in. The summary's write and
  a purge now share a lock, and the write goes ahead only if the raw log is still there;
  otherwise the result is dropped (`Purged`) and the learner logs it at INFO.
- **A purge removes chunks whose file is already gone.** It deleted the chunks only of
  files that still existed, so chunks left under a missing log's name survived every purge.
  It now deletes both sources' chunks unconditionally, and reports a purge if it removed
  either files or chunks.
- **No model call goes out once a session starts.** The rule was checked only between
  sessions and before each campaign merge. A long session's windows and merge all went out
  after a session had started, which the startup catch-up made the normal case. Every
  summary call now asks first and, if a session is active, stops with nothing written
  (`Paused`); the learner puts that session back at the front of the queue.
- **A session learnt during a catch-up listing is not learnt again** (`learning.py`).
  `catch_up` lists the logs outside the lock, so one finished in between was queued again
  and summarised twice. The worker now skips a queued log whose summary exists.
- **The summary's store update is all or nothing.** The old chunks were deleted and the
  file written before the new chunks were embedded, so an embedder failure left a summary
  file (the session counted as learnt, never retried) and nothing in the store.
  `Store.add_chunks` takes `replace=`: it embeds first, then deletes and inserts in one
  transaction. The summary file is written last.
- **Short lines are not folded into a question about them** (`transcript.py`). A later line
  repeating 60 percent of an earlier one replaced it, so "The innkeeper is called Bram."
  followed by "Who is the innkeeper called again?" lost the name. A line of under six
  words is now replaced only by a line that begins with all of it, give or take one word
  gained at the front. All but one short fold in the 2026-09-16 log are of that shape and
  still fold; the exception is a lone "I", now kept as a line of its own (39 rows and 10068
  characters de-duplicated, from 38 and 10067). Longer lines keep the 60 percent rule, so a long statement followed by a long
  question that repeats most of it can still fold; on that log only 1 of 32 folds was more
  than 30 s apart.
- **The campaign memory's cap keeps every section** (`campaign.py`). It cut from the end,
  and Story so far comes first, so an over-long reply lost NPCs, Places, Open threads, Loot
  and Decisions for good. It now drops the oldest Story so far lines, keeping the latest;
  a reply that still does not fit is refused and the memory stays due.
- **Campaign chunks no longer fill every sessions slot.** At weight 2.0 they took all three
  for almost any question, and a fact that only a session summary kept never reached the
  prompt or `recall`. Measured with the BGE embedder: at 2.0, "what was the seal on the
  letter?", "what did you say about the lotus?" and "what did the courier look like?" each
  returned three campaign chunks; at 1.6 the summary's Loot, Memorable lines and NPCs met
  chunks each made the top three, and "where were we?" and "what should we do next?" still
  went to the memory. `CAMPAIGN_WEIGHT` is now 1.6, above a summary's 1.5 and below a
  deliberate note's 2.0. Sections that are only "- none" are no longer indexed.
- **A reply cut off at `max_tokens` is a failure** (`llm.py`). `LLMClient.complete` accepted
  it as whole, so a summary or memory without its last headings was written and never
  retried. It now raises when `finish_reason` is `length`. Because that failure would
  repeat on every retry, merges get `MERGE_MAX_TOKENS` 2000 (a merge is then under about
  10.2k tokens at 3 characters a token) and the campaign merge 2500 (a memory at the 6,000
  character cap is about 2000 tokens).
- **The prompt says the notes outrank the memory of play** (`knowledge.py`). The merge
  prompt said the notes win, but the notes are never passed to the merge, so that clause is
  gone. When any sessions line reaches the "What you know" block, a closing line now says
  that `[notes]` are the DM's canon and `[sessions]` only the familiar's memory of play.
- **Sessions is not squeezed out of the prompt.** `build_context` read the corpora in turn
  and stopped at the first line over budget, so sessions, read last, was cut first once
  long rules were indexed. Each corpus's best hit now goes in before any corpus's second,
  a line that does not fit is skipped rather than ending the block, and the lines are
  still written in corpus order.

Not run against the real stack; the retrieval weights were measured with the real BGE
embedder on a scratch store. The purge route still allows a purge of the session the
learner holds; the lock makes that safe.

New tests: `test_a_purge_landing_mid_summary_leaves_nothing_behind`,
`test_no_call_goes_out_once_a_session_starts_mid_summary` and
`test_a_session_learnt_while_catch_up_listed_it_is_not_learnt_again` in
`tests/test_learning.py`; `test_a_purge_forgets_chunks_whose_file_is_already_gone` and
`test_an_embedder_failure_leaves_the_live_chunks_and_no_summary` in
`tests/test_summariser.py`; `test_a_short_line_echoed_by_a_later_one_is_kept` in
`tests/test_transcript.py`; `test_an_over_cap_memory_keeps_every_heading_and_open_thread`
and `test_a_session_summary_fact_is_still_retrievable_once_a_campaign_memory_exists` in
`tests/test_campaign.py`; `test_complete_refuses_a_reply_cut_off_at_max_tokens` in
`tests/test_llm.py`; `test_the_context_says_the_notes_outrank_the_memory_of_play` and
`test_long_rules_do_not_squeeze_out_the_sessions_corpus` in `tests/test_knowledge.py`.
`test_it_is_indexed_in_sessions_under_its_own_source_at_the_top_weight` is renamed
`..._just_above_a_summary` and checks 1.6. `tests/fakes/llm_server.py` can now script
`finish_reason`.
