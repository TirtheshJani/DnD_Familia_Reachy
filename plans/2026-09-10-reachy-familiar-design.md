# Reachy Mini D&D Familiar: Design Spec

Date: 2026-09-10
Status: approved design, ready for writing-plans
Owner: TJ (Tirthesh Jani)
Campaign: Shadows over Nephria (in person, 5e, Art Deco noir setting)
Target executor: Claude Code (Opus 5 Max)

## 0. Instructions for Claude Code

Read this whole file before touching anything. Then invoke the `writing-plans` skill to turn it into an implementation plan, present the plan for sign-off, and only then invoke `executing-plans`. Use `test-driven-development` for every feature, `systematic-debugging` before any fix, and `verification-before-completion` before claiming anything works. Never claim success without running the code.

Coding conventions: Python 3.11+, Flask for the brain service, React 19 with TypeScript and Vite for the companion UI, `uv` for Python environments. Follow the Karpathy guidelines: surgical changes, surface assumptions, no speculative abstractions. Prose in docs uses British spelling and no em or en dashes.

Before writing code, read `https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md`, the conversation app README at `https://github.com/pollen-robotics/reachy_mini_conversation_app`, and the speech-to-speech README (`pip show speech-to-speech` for the version, then the matching GitHub tag). Pin every version you install in `pyproject.toml` and record them in `docs/versions.md`. The blog posts this design was researched from are dated May 27 and June 3, 2026; assume flags and tool APIs may have shifted since and verify against the installed version, not against this document.

## 1. Purpose

A Reachy Mini Wireless that plays TJ's character's familiar at an in-person D&D table. It listens to the whole table continuously, speaks only when addressed, reacts non-verbally to the game, rolls dice, looks up rules, remembers campaign notes and past sessions, and never leaves the room's network. The creature concept is undecided until Session Zero, so the persona is a swappable config file.

Success looks like: a full four-hour session where the familiar answers every line addressed to it within 1.5 seconds, speaks zero times when not addressed, never talks over the DM, and produces a usable session summary afterwards.

## 2. Non-goals for v1

No voice cloning. No multiple personas active at once. No code running on the CM4 beyond the stock conversation app. No cloud calls of any kind, including embeddings and vision. No remote MCP Spaces. No player identification or face recognition: the familiar tracks faces and looks at people, it never learns who they are.

## 3. System overview

Five processes. One on the robot, four on the Zephyrus G14 (Ryzen 9 7940HS, 16 GB RAM, RTX 4080 Laptop 12 GB VRAM, Windows 11) inside WSL2 Ubuntu with CUDA passthrough.

```
Reachy Mini Wireless (CM4)
  reachy_mini_conversation_app, profile "familiar"
  connection: ws://<zephyrus-lan-ip>:8765/v1/realtime
        |  audio in / audio out / tool calls
        v
Zephyrus, WSL2
  [A] speech-to-speech        realtime mode, Silero VAD, Parakeet-TDT 0.6B v3, Qwen3-TTS
        |  Responses API (HTTP, streaming)
        v
  [B] familiar-brain (Flask)  gate, retrieval, tools, persona, session log
        |  Responses API (HTTP, streaming)
        v
  [C] llama-server            Gemma 4 E4B, 2 slots, flash attention
  [D] companion UI            React/Vite, served by [B] on LAN for phone access
```

Speech-to-speech is configured with `--responses_api_base_url` pointing at [B], not at [C]. [B] is a Responses API proxy that adds behaviour. The official app and the official voice engine are used unmodified.

## 4. Component specs

### 4.1 Robot profile: `familiar`

A conversation app profile directory containing `instructions.md` (a stub; the real system prompt is injected by the brain), `tools.txt`, and any profile metadata the installed app version requires.

`tools.txt` enables built-ins `move_head`, `play_emotion`, `stop_emotion`, `idle_do_nothing`, `head_tracking`, `camera`, and the local custom tools listed in 4.5. It does not enable `dance` or any remote MCP tool. See 4.7 for how camera and tracking are used.

Verify in the installed version how custom local tools are discovered (the README describes one Python module per tool, subclassing `Tool` with `name`, description, `parameters_schema`, and async `__call__`). Tool modules that only need the model's decision (dice, lookup, recall, note, initiative) are executed inside the brain, not on the robot; the robot-side tool module is a thin forwarder if the app requires tool execution to happen on its side. Determine this during planning by reading the app's tool dispatch code, and document the finding.

### 4.2 [A] speech-to-speech

Installed with `uv pip install speech-to-speech` inside WSL2. Launch command (verify flags against the installed version):

```
speech-to-speech \
  --mode realtime \
  --host 0.0.0.0 --port 8765 \
  --stt parakeet-tdt \
  --tts qwen3 \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --qwen3_tts_speaker <from persona.yaml> \
  --qwen3_tts_backend ggml \
  --llm_backend responses-api \
  --model_name familiar \
  --responses_api_base_url http://127.0.0.1:8090/v1 \
  --responses_api_api_key "" \
  --responses_api_stream \
  --enable_live_transcription \
  --thresh <vad threshold, start 0.6, tune at the table> \
  --chat_size 12
```

The brain owns conversation history, so `--chat_size` is kept small and the brain reconstructs context per turn. If Qwen3-TTS 1.7B plus everything else does not fit in 12 GB, fall back to the 0.6B CustomVoice model. Record the measured VRAM in `docs/versions.md`.

### 4.3 [C] llama-server

```
llama-server -hf ggml-org/gemma-4-E4B-it-GGUF -np 2 -c 32768 -fa on --swa-full --port 8080 --host 127.0.0.1
```

Two slots: slot one for the spoken response, slot two for the gate classifier so the classifier never waits behind generation. Context reduced from the blog's 64k to 32k to save VRAM; the brain keeps prompts short. Fallback model if latency misses the 2.5 s ceiling: `Qwen/Qwen3-4B-Instruct-2507` GGUF at Q4_K_M.

### 4.4 [B] familiar-brain

Flask app, package name `familiar_brain`, port 8090. Single-process, async where the installed Flask supports it, otherwise use threads; the gate classifier and generation must be able to run concurrently.

Endpoints:

- `POST /v1/responses` implements enough of the Responses API for speech-to-speech: accepts the input items, streams SSE output events, supports tool call and tool result items. Determine the exact event set speech-to-speech consumes by reading its `responses-api` backend source and implement only that. Write a contract test against a recorded fixture of a real speech-to-speech request.
- `GET /api/state` returns mute state, listening state, current session id, last transcript segments, and last turn timings.
- `POST /api/mute`, `POST /api/unmute`, `POST /api/mute_and_stop_logging`.
- `POST /api/speak_now` forces the next (or last, if within 5 s) segment through the gate as ADDRESSED.
- `POST /api/note` appends a deliberate note to the session log.
- `POST /api/session/start`, `POST /api/session/end` (end triggers the summariser job), `DELETE /api/session/<id>` (purge).
- `GET /` serves the built companion UI.

Internal modules, each independently testable:

`persona.py` loads `persona.yaml`, validates it, and renders the system prompt. Fixed blocks are appended by code and cannot be overridden by the file: the off-limits list, the "never speak for the DM or resolve disputes" rule, and the two-sentence ceiling.

`gate.py` decides ADDRESSED, AMBIENT, or IGNORE per transcript segment. See section 5.

`retrieval.py` wraps SQLite plus sqlite-vec and a local embedding model. Corpus tag per row. `search(query, corpus, k)` returns chunks with scores. See section 6.

`tools/` one module per tool. See 4.5.

`session.py` owns the session log (JSONL, one line per segment with timestamp, speaker guess if available, text, gate decision, and whether the familiar spoke), plus the post-session summariser job.

`llm.py` is the only module that talks to llama-server. Two functions: `generate_stream(messages, tools)` on slot one and `classify(prompt)` on slot two with `max_tokens` capped at 4.

`motion.py` maps persona moods to built-in emotion names and emits the corresponding tool call items in the response stream.

`timing.py` writes one structured log line per turn: `gate_ms`, `retrieval_ms`, `first_token_ms`, `generation_ms`, `total_ms`, decision, and whether speech happened.

### 4.5 Tools exposed to the model

`roll(expression: str, advantage: "none"|"adv"|"dis" = "none")` parses standard dice notation (`1d20+5`, `2d6kh1`, `4d6dl1`) and returns total plus breakdown. The tool does all arithmetic. Pure function, fully unit tested including malformed input.

`lookup_rule(query: str)` searches the rules corpus, returns up to 3 chunks with section titles.

`recall(query: str, scope: "notes"|"sessions"|"all" = "all")` searches notes and transcript corpora.

`note(text: str)` appends to the session log tagged `deliberate`.

`initiative(action: "add"|"next"|"clear"|"show", name: str | None, value: int | None)` maintains an in-memory turn order for the current session, persisted to the session log.

Built-ins used: `play_emotion`, `stop_emotion`, `move_head`, `idle_do_nothing`, `head_tracking`, `camera`.

### 4.6 [D] Companion UI

React 19, TypeScript, Vite, single page, phone-first layout. Four large controls: Mute, Unmute, Speak Now, and a note field with send. A status strip shows listening state, session timer, last three transcript lines, and last turn latency. A settings drawer has: end session, purge session, gate follow-up window seconds, and VAD threshold display (read-only, the flag lives in [A]). Built output is copied into `familiar_brain/static/` and served by Flask. No auth; LAN only. Use the `frontend-design` skill for the visual layer.

### 4.7 Vision and liveliness

The goal is a creature that visibly attends to the room, not a camera that answers questions. Three behaviours, in priority order.

Face tracking. The conversation app's built-in `head_tracking` tool and its face-tracking motion layer run on the robot and blend with speech wobble and idle breathing. The brain drives when it is on: enable tracking whenever the gate returns ADDRESSED or during the follow-up window (look at whoever is talking to you), keep it on while speaking, and hand back to idle behaviour a few seconds after the exchange ends. During long stretches of IGNORE the brain runs a slow "attend" routine: every 20 to 40 seconds (randomised) it either re-enables tracking briefly to find a face and settle on it, or issues a small `move_head` glance toward the last known face position, so the creature reads as watching the table rather than staring at one spot. AMBIENT decisions may trigger a glance toward the speaker. If the SDK exposes sound direction from the microphone array, use it to pick the glance direction; verify during planning and fall back to last-known-face if not.

Camera as a sense, on demand. When addressed with a question that needs sight ("what do you see", "how many of us are here", "read the card"), the model calls the built-in `camera` tool. The frame must be described locally. Two options, decided during planning: if the Gemma 4 E4B GGUF ships a multimodal projector that llama-server supports, the brain forwards the image item to slot one and no extra model is needed; otherwise the brain runs `HuggingFaceTB/SmolVLM2-2.2B-Instruct` on the Zephyrus GPU behind a small internal endpoint and injects the caption as text. Do not use the conversation app's `--local-vision` flag, since on the Wireless that would run the VLM on the CM4. Frames are processed in memory and never written to disk or the session log; only the text description is logged.

Reactive glances. `motion.py` gains a `glance_at_speaker` action used by AMBIENT and by the attend routine. It is a `move_head` toward a bearing, clamped to the app's safety limits.

VRAM impact: SmolVLM2 2.2B adds roughly 2 GB if needed. If the budget does not close with Qwen3-TTS 1.7B, drop TTS to 0.6B before dropping vision.

Consent: the camera sees other players. The companion UI mute state also disables `camera` (tool calls return "eyes closed"), and the physical mute cue covers both senses. Raise camera use at Session Zero alongside listening.

## 5. Address gate

Evaluated in order. Mute check first, then the three layers. First layer to return a decision wins.

Layer 0, mute: if muted, return IGNORE. If "mute and stop logging", drop the segment before logging.

Layer 1, deterministic: segment contains the persona name or any alias (case-insensitive, word boundary), or `speak_now` is armed. Returns ADDRESSED.

Layer 2, follow-up window: if the familiar finished speaking fewer than N seconds ago (default 8, configurable), return ADDRESSED.

Layer 3, classifier: one call to `llm.classify` on slot two with the last four segments, the persona name, and a fixed instruction to answer exactly one token from ADDRESSED, AMBIENT, IGNORE. Timeout 600 ms. On timeout, error, or unparseable output, return IGNORE.

Decision effects: ADDRESSED runs retrieval and generation and speaks. AMBIENT may emit one `play_emotion` or `move_head` chosen by a small keyword rule set in `motion.py` (ominous words, crit, natural one, laughter) and never speaks. IGNORE returns a response containing only `idle_do_nothing`.

Physical state cues via built-in motions: muted is antennas down and still; listening is idle sway; gate opening is a quick antenna flick before speech begins.

## 6. Knowledge layer

Storage: one SQLite file `data/familiar.db` with sqlite-vec. Tables `chunks(id, corpus, source, title, text, embedding, created_at)` and `sessions(id, started_at, ended_at, summary_path, raw_retained_until)`.

Embedding model: `BAAI/bge-small-en-v1.5` or `nomic-ai/nomic-embed-text-v1.5`, whichever installs cleanly under WSL2 CUDA; pick one during planning and record it. Runs local, never cloud.

Corpus `rules`: `DnD_BasicRules_2018.pdf` chunked by section heading using the pdf-reading skill approach, indexed once by a CLI command `familiar-brain index rules <pdf>`.

Corpus `notes`: a watched folder `data/notes/` of markdown files. Re-index on change using a file watcher; chunk by heading.

Corpus `sessions`: per session, the raw JSONL log and the post-session summary markdown. Summary is produced by a background job after `session/end`, using Gemma with a fixed prompt yielding sections: scenes, NPCs met, decisions, loot, open threads, memorable lines. Summary chunks are weighted 1.5x over raw log chunks at query time. Raw logs are deleted after 7 days by default; summaries are kept. Both values are config. Purge deletes both immediately.

Retrieval runs only on ADDRESSED. Cap 3 chunks per corpus. Total retrieved text injected into the prompt is capped at roughly 1,200 tokens.

## 7. Persona file

`config/persona.yaml`:

```yaml
name: Understudy
aliases: [understudy, little one]
tts_speaker: <pick a neutral Qwen3-TTS built-in speaker>
manner: |
  Attentive and dry. Curious about mortals and their habits.
  Never louder than the table. Answers in one or two sentences.
gestures:
  amused: play_emotion:<name>
  alarmed: play_emotion:<name>
  thoughtful: move_head:<pose>
  proud: play_emotion:<name>
follow_up_window_s: 8
```

Populate gesture values from the installed app's emotion library at implementation time. Fixed blocks appended by code, not editable here: the off-limits list (explicit sex, explicit harm to children or animals, heatstroke, homophobia, overt real-world religion, pregnancy horror, racism, self harm, sexism, sexual assault, spiders, and the character names Amber, Bella, Dana, Mike), the DM rule, and the length ceiling.

## 8. Error handling

Every failure degrades to silence plus a visible cue, never to a stuck robot.

- llama-server unreachable: return `idle_do_nothing`, emit the "droop" motion, log error.
- Retrieval failure: proceed without context, log warning.
- Classifier timeout: IGNORE.
- speech-to-speech WebSocket drop: nothing to do in the brain; it is stateless per turn. The conversation app reconnects.
- Tool exception: return a tool result string describing the failure so the model can say "my dice slipped" rather than crashing the turn.
- Session log write failure: keep the turn going, surface a red badge in the companion UI.

## 9. Testing

Unit: dice parser (property-based tests for expression grammar), gate layers 0 to 2, persona loader and fixed-block injection, chunkers, off-limits filter, initiative tracker.

Integration: brain against a fake llama-server (`tests/fakes/llm_server.py`) and a recorded fixture of a mock session transcript (`tests/fixtures/mock_session.jsonl`) with DM narration, cross-talk, two addressed lines, one follow-up, one ambient trigger. Assert exactly which segments produce speech, which produce motion only, and which are silent. A contract test replays a captured real speech-to-speech request against `/v1/responses` and validates the SSE stream shape.

Hardware smoke checklist (`docs/smoke-test.md`, run before the first session): mute cue visible, wake by name, dice roll spoken correctly, rule lookup, follow-up window, Speak Now from phone, head turns to the speaker when addressed, attend routine glances during silence, camera question answered locally, mute closes the eyes, end session produces a summary file.

Latency: target 1.5 s end of speech to first audio on ADDRESSED turns, ceiling 2.5 s, measured from `timing.py` lines over the smoke test. If the ceiling is missed, swap to Qwen3-4B before touching anything else.

## 10. Repository layout

```
reachy-familiar/
  pyproject.toml
  README.md
  docs/
    versions.md
    smoke-test.md
    superpowers/specs/2026-09-10-reachy-familiar-design.md   (this file)
  config/
    persona.yaml
    persona.example.yaml
    brain.toml            ports, paths, retention, window, thresholds
  familiar_brain/
    __init__.py  app.py  persona.py  gate.py  retrieval.py  session.py
    llm.py  motion.py  timing.py  cli.py
    tools/  roll.py  lookup_rule.py  recall.py  note.py  initiative.py
    static/               built companion UI
  robot_profile/familiar/ instructions.md  tools.txt  (plus forwarder tool modules if required)
  companion/              React/Vite source
  scripts/
    start_all.sh          launches [C], [A], [B] in order with health checks
    index_rules.sh
  data/                   gitignored: familiar.db, notes/, sessions/
  tests/
```

## 11. Milestones (for writing-plans to expand)

1. Environment: WSL2 CUDA verified, speech-to-speech and llama-server running with the stock realtime loop, robot talking to the laptop with the default profile. Record versions and VRAM.
2. Brain skeleton: `/v1/responses` passthrough proxy that changes nothing, contract test green, robot still talks.
3. Persona and fixed blocks. Session logging and timing lines.
4. Gate layers 0 to 2 with unit tests; layer 3 classifier on slot two; integration test on the mock session.
5. Dice, initiative, note tools.
6. Retrieval layer, rules index, notes watcher, `lookup_rule` and `recall`.
7. Post-session summariser and retention job; purge.
8. Companion UI.
9. Motion mapping and physical state cues.
10. Vision: face tracking control, attend routine, camera path with local VLM, mute covers camera.
11. Hardware smoke test, latency measurement, tuning of VAD threshold and follow-up window.

## 12. Open items to resolve during planning, not before

Exact Responses API event subset speech-to-speech consumes. Whether custom tools execute robot-side or can be satisfied entirely in the brain. Which Qwen3-TTS speaker to use for the placeholder. Whether the installed app exposes an antenna-only motion for the mute cue or whether `play_emotion` names must be reused. Embedding model choice under WSL2 CUDA. Whether Gemma 4 E4B GGUF has a working multimodal projector in llama-server, else SmolVLM2 on the Zephyrus. Whether the SDK exposes sound direction from the mic array. How `head_tracking` on and off is toggled from a tool call in the installed app version.
