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
