# Rocky Audio Stall and Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rocky keeps hearing and answering for a whole D&D session (Thu 2026-11-26 night), and when any part of the audio path does fail it recovers by itself within seconds and leaves logs that say why.

**Architecture:** Three layers of defence. First, evidence: every launch keeps its own logs. Second, root-cause fixes for the silent, permanent failures the 2026-11-26 trace found and reproduced (native mic thread death, an unanswered robot tool call blocking s2s forever, unbounded LLM waits, a wedged s2s pool). Third, watchdogs so that anything we have not found turns into a loud restart instead of silence. Polish fixes for the Companion UI and speaker ID follow once the audio path is solid.

**Tech Stack:** Python 3.11+, Flask (familiar-brain), asyncio (AudioTapProxy), HF speech-to-speech 1.0.0 patched through `scripts/run_s2s.py`, stdlib WebSocket client in `reachy_familiar`, bash (`scripts/start_all.sh`) under WSL2, pytest.

**Spec:** No separate spec. This plan argues from the investigation findings, saved alongside it at `docs/superpowers/plans/2026-11-26-rocky-audio-stall-findings.md` (ranked causes with file:line evidence, repro output and verifier verdicts).

**Execution order (TJ confirmed 2026-11-26 that Rocky ran Pollen's official conversation app, started from the Reachy Mini desktop app):** Task 1, then Task 4 (the leading cause on this path), Task 5, Task 6, Task 9, Task 7, Task 8, then the native-app Tasks 2 and 3 last (fallback path only). Task 10 runs after all merged.

## Global Constraints

- Do not edit installed packages (s2s, reachy_mini, Pollen app) in place. Patch s2s only through `scripts/run_s2s.py` `apply_s2s_patches()`; robot-side changes are runbook steps, not code here.
- Every new failure path logs at WARNING or above. No `log.debug` in an `except` that ends a loop or thread.
- Tests run with `uv run pytest -q`; pytest-timeout is 20 s per test (pyproject), so every socket or thread test uses explicit timeouts under that.
- Baseline before any change: 766 pass, 25 fail, 10 skip in a clean checkout without local data. All 25 failures come from gitignored local data (`data/notes/`, `config/persona.yaml`, `data/sessions/`). On TJ's laptop the expected baseline is whatever `uv run pytest -q` reports before Task 1; record it in the first commit message and do not let it get worse.
- Log one entry per task in `docs/deviations.md`, as the repo already does.
- No em dashes in any user-facing copy or docs.
- Live robot steps happen on TJ's laptop and robot only. They are listed in Task 10 and are not done by subagents.

## Review Focus

1. A stall shorter than the s2s ping timeout (about 5 to 20 s) on Wi-Fi or WSL2: Rocky should keep or regain hearing within 5 s of the network recovering, with no button press. Owned by Task 2 (native) and Task 4 (tool-call gate), pinned by their stall tests.
2. Rocky's reply takes more than 30 s (cold Gemma, GPU contention): the next voice turn must still be answered. Owned by Task 4 and Task 5, pinned by `test_unanswered_gesture_call_does_not_block_next_turn` and `test_llm_stream_has_total_deadline_and_no_retries`.
3. One stack process dies or wedges mid-session (s2s, tap, brain, llama): the launcher must notice within 10 s, say which one in the log, and either restart it or exit non-zero so the wake daemon shows an error. Owned by Task 6.
4. A player talks for longer than 25 s without a pause: the speech is transcribed, not silently dropped. Owned by Task 5, pinned by `test_start_all_does_not_set_finite_max_speech_ms`.
5. TJ presses Reconnect in the laptop Companion UI while the robot is stuck: it must actually reconnect the robot for whichever robot app is running. Owned by Task 7, pinned by `test_app_js_has_no_undeclared_state_refs` and `test_robot_reconnect_route_reaches_native_app`.

---

### Task 1: Keep every launch's logs

The failing run on 2026-11-25 left four empty logs because `start_all.sh` truncates them with `>` on every launch, and the wake daemon's stack output goes to a `pythonw.exe` stderr that nobody sees. Nothing else in this plan can be judged live without this.

**Files:**
- Modify: `scripts/start_all.sh:142,148,150,173,175,191` (log redirections), plus a new `RUN_DIR` block near the top
- Modify: `familiar_brain/wake_daemon.py` (logging setup and `_default_spawn`, around lines 119-135)
- Test: `tests/test_start_all_logs.py` (new), `tests/test_wake_and_reachy_app.py`

**Interfaces:**
- Produces: `data/logs/<YYYYmmdd-HHMMSS>/{llama,brain,s2s,audio_tap,robot_connect}.log`, and `data/logs/latest` as a symlink to the newest run dir. The old `data/*.log` paths become symlinks into the current run dir so existing docs and `wait_for` tails still work. Env var `RUN_DIR` exported to children.
- Produces: wake daemon log at `data/logs/wake_daemon.log` (rotating, 5 MB x 3) and stack stdout appended to `$RUN_DIR/stack.log`.

- [ ] **Step 1: Write failing tests.** `test_start_all_keeps_previous_run_logs`: run `start_all.sh` twice with stub binaries on PATH (stubs print one line and exit 0; use the existing `--skip-robot --no-audio-tap` flags and `STARTUP_TIMEOUT_S=3`), assert two run dirs exist under `data/logs/` and each contains a non-empty `brain.log`. `test_wake_daemon_logs_to_file`: construct the daemon with `tmp_path` as repo root, call its logging setup, log one line, assert `data/logs/wake_daemon.log` contains it.
- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/test_start_all_logs.py tests/test_wake_and_reachy_app.py -k "logs" -v`. Expected: FAIL (single truncated log, no file handler).
- [ ] **Step 3: Implement.** In `start_all.sh`, create `RUN_DIR="data/logs/$(date +%Y%m%d-%H%M%S)"`, redirect each child with `>> "$RUN_DIR/<name>.log" 2>&1`, then `ln -sfn` the `latest` link and the legacy `data/<name>.log` links. Also prefix each line with a timestamp where the child does not (s2s does; for the others pipe through `ts` only if available, otherwise skip). Keep only the newest 20 run dirs. In `wake_daemon.py`, add a `RotatingFileHandler` and append the spawned stack's stdout to `$RUN_DIR/stack.log`.
- [ ] **Step 4: Run the tests and confirm they pass,** then the full suite. Expected: PASS, baseline unchanged.
- [ ] **Step 5: Commit.** `git commit -m "fix(launch): keep per-run logs instead of truncating them"`

### Task 2: Native app never goes deaf after a send stall

Reproduced cause. `SimpleWebSocket.recv_text` sets a 0.5 s timeout on the socket the mic thread also uses for `sendall`, so any send stall over 0.5 s raises `TimeoutError`; `_mic_loop` logs at DEBUG and exits while `_run_session` keeps answering pings with `connected=True`. Verified on Linux that the timed-out send writes no partial frame, so nothing else ever notices. The same per-call timeout also tears the session down when a server frame arrives split across more than 0.5 s.

**Files:**
- Modify: `reachy_familiar/reachy_familiar/audio_bridge.py:83-87` (`SimpleWebSocket.__init__`), `:144-170` (`recv_text`), `:358-381` (`_run_session` loop), `:387-425` (`_mic_loop`)
- Test: `tests/test_audio_bridge_backpressure.py` (new)

**Interfaces:**
- Produces: `SimpleWebSocket.recv_text(timeout: float = 1.0) -> str | None` with unchanged signature and new semantics: `timeout` only bounds the wait for the first byte of a frame; once a frame starts it is read with the socket's long timeout (10 s). The socket timeout is set once in `__init__` and never changed afterwards.
- Produces: `SimpleWebSocket.close() -> None` safe to call from any thread (shutdown then close, swallow `OSError`).
- Produces: `RealtimeAudioBridge.mic_alive: bool` (True while the `reachy-mic-tx` thread runs), for the heartbeat in Task 3.

- [ ] **Step 1: Write the failing tests.**
  - `test_mic_thread_survives_or_reconnects_after_send_stall`: a fake WS server thread with `SO_RCVBUF=8192` does the handshake, reads for 1 s, stops reading for 4 s, then reads again and pings every 1 s. FakeMedia returns a `(4096, 2)` float32 array every 10 ms. Run `RealtimeAudioBridge.run_loop` in a thread. Four seconds after the stall ends assert: a thread named `reachy-mic-tx` is alive or the server accepted a second connection; the server read at least one `input_audio_buffer.append` after the stall on the current connection; and `bridge.connected` is False whenever no mic thread is alive. Whole test under 15 s.
  - `test_partial_frame_does_not_tear_down_session`: fake server sends the first 3 bytes of a text frame, sleeps 0.8 s, sends the rest. Assert `recv_text(timeout=0.5)` returns the full text (it may return None on the first call before any byte arrives, but must not raise).
  - `test_mic_send_failure_triggers_reconnect`: fake WS whose `send_json` raises `TimeoutError` on the 5th call. Assert `_run_session` exits within 2 s and a WARNING containing "mic" was logged (caplog).
- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/test_audio_bridge_backpressure.py -v`. Expected: the first and third FAIL (mic thread dead, connected stays True), the second FAILS with `socket.timeout`.
- [ ] **Step 3: Implement.** In `recv_text`, replace `self.sock.settimeout(timeout)` with a `select.select([self.sock], [], [], timeout)` check that returns None only when `self._buf` is empty and nothing is readable. In `_run_session`, after each `recv_text`, raise `ConnectionError("mic sender thread exited")` if the mic thread is dead and `stop_event` is not set. In `_mic_loop`'s except branch, log at WARNING and call `ws.close()` before `break`.
- [ ] **Step 4: Run the tests and confirm they pass,** then `tests/test_native_robot_and_profiles.py`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "fix(native): mic thread survives send stalls and reconnects when it dies"`

### Task 3: Native app restarts dead mic capture and reports it

If GStreamer or ALSA stops delivering samples, `get_audio_sample()` returns None forever and `_mic_loop` sleeps in a loop with no restart. Capture is also started only once in `run_loop`, so a reconnect never re-arms it.

**Files:**
- Modify: `reachy_familiar/reachy_familiar/audio_bridge.py:288-297` (move start calls), `:387-399` (`_mic_loop` None branch)
- Modify: `reachy_familiar/reachy_familiar/main.py:201,228` (heartbeat fields)
- Test: `tests/test_audio_bridge_backpressure.py`

**Interfaces:**
- Consumes: `RealtimeAudioBridge.mic_alive` from Task 2.
- Produces: `RealtimeAudioBridge.mic_stalled: bool`, and heartbeat/boot-status fields `mic_alive` and `mic_stalled`.

- [ ] **Step 1: Write the failing test** `test_mic_loop_restarts_dead_capture`: FakeMedia returns `np.full((320, 2), 0.01, np.float32)` for 0.5 s, then None until `start_recording` is called after a `stop_recording`, then frames again. FakeWS records `send_json` types. Run `_mic_loop` for 3.5 s. Assert `stop_recording` and `start_recording` were each called within 2.5 s of the stall, a WARNING containing "stalled" was logged, and appends resumed after the restart. Second test `test_capture_dead_after_two_restarts_forces_reconnect`: media never recovers; assert `ws.close()` was called.
- [ ] **Step 2: Run and confirm it fails.** Expected: FAIL (no restart).
- [ ] **Step 3: Implement.** Track `last_sample_at`; after 1.5 s of None, log WARNING, `stop_recording()`, `start_recording()`, `start_playing()`; after 2 failed restarts, set `mic_stalled` and `ws.close()`. Move the initial start calls to the top of `_run_session` (after a `stop_recording()`), and add both flags to the heartbeat.
- [ ] **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit.** `git commit -m "fix(native): restart stalled mic capture and surface it in the heartbeat"`

### Task 4: An unanswered robot tool call can no longer block s2s

Verified in code and by running it. The brain sends a robot tool call (gesture preface or `idle_do_nothing`) on nearly every turn. s2s refuses every later LLM request while any `function_call_output` is pending, with no timeout. The official app drops its tool result if `response.done` takes more than 30 s, so one slow reply makes Rocky silent until reconnect. Transcription keeps arriving but nothing answers, which matches "it stops working".

**Files:**
- Modify: `scripts/run_s2s.py` (new `patch_chat_pending_tools`, called from `apply_s2s_patches()` at line 220)
- Modify: `familiar_brain/app.py:774` (non-addressed turns)
- Test: `tests/test_run_s2s.py`, `tests/test_app_integration.py`

**Interfaces:**
- Produces: `patch_chat_pending_tools(chat_cls: type[Any]) -> bool` and module constant `ROBOT_GESTURE_TOOLS = frozenset({"play_emotion", "stop_emotion", "head_tracking", "move_head", "idle_do_nothing"})`. Patched `Chat.has_pending_tool_calls()` returns True only for pending calls whose name is not in that set. Late outputs for gesture calls are still accepted.

- [ ] **Step 1: Write the failing tests.** `test_unanswered_gesture_call_does_not_block_next_turn`: after `apply_s2s_patches()`, build a `Chat`, add a user message, a `function_call(name="play_emotion", call_id="call_x")` with no output, then a second user message; assert `chat.has_pending_tool_calls()` is False. Assert it is True for a pending call named `camera`. Assert a late `function_call_output` for `call_x` is accepted without raising. Skip with a clear reason if `speech_to_speech` is not importable (follow how the file already handles that). `test_ignored_turn_sends_no_tool_call`: POST a non-addressed line to the brain's `/v1/responses` and assert the SSE body has no `function_call` item.
- [ ] **Step 2: Run and confirm they fail.** Expected: first FAILS (pending blocks), second FAILS (`idle_do_nothing` emitted).
- [ ] **Step 3: Implement** the patch (hold `self._lock`, as the original does), and at `app.py:774` send `sse` of `empty_response_events()` (already in `familiar_brain/responses_api.py:152`) instead of `tool_call_events(idle_items())` when there is no preface.
- [ ] **Step 4: Run and confirm pass,** plus `tests/test_responses_api.py`.
- [ ] **Step 5: Commit.** `git commit -m "fix(s2s): unanswered gesture calls no longer block later turns"`

### Task 5: Bounded LLM turns and no dropped long speech

The s2s LLM thread can block for about 60 s on one brain request (httpx 20 s per read, two SDK retries, and a brain that sends no headers until llama's first token), and nothing can interrupt it. `--max_speech_ms 25000` throws away any segment over 25 s with no transcript.

**Files:**
- Modify: `scripts/run_s2s.py` (new `patch_responses_request`, registered in `apply_s2s_patches()`)
- Modify: `familiar_brain/app.py:780` (`gen()` yields `": ok\n\n"` first), `familiar_brain/upstream.py:21,64` (timeout to `(5, 15)`)
- Modify: `scripts/start_all.sh:168` (remove `--max_speech_ms 25000`)
- Test: `tests/test_run_s2s.py`, `tests/test_start_all_vad_args.py` (new), `tests/test_upstream.py`

**Interfaces:**
- Produces: `patch_responses_request(lm_cls: type[Any]) -> bool`; env var `FAMILIAR_LLM_DEADLINE_S` (default 20). On deadline it closes the stream and raises `httpx.ReadTimeout`, so s2s's existing fallback path ends the turn.

- [ ] **Step 1: Write the failing tests.** `test_llm_stream_has_total_deadline_and_no_retries`: a local `ThreadingHTTPServer` `/v1/responses` sends one `response.output_text.delta` every 0.5 s forever and counts POSTs; with `FAMILIAR_LLM_DEADLINE_S=1.5` the patched iterator raises within 3 s and the server saw exactly 1 POST. `test_start_all_does_not_set_finite_max_speech_ms`: parse `S2S_ARGS` in `start_all.sh` and assert `--max_speech_ms` is absent. `test_addressed_stream_sends_headers_immediately`: with a stub upstream that sleeps 2 s before its first event, the brain's first SSE bytes arrive in under 0.5 s.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.** Wrap `_request` to use `client.with_options(max_retries=0)`; wrap `_iter_stream_events` with a `threading.Timer` that calls `api_response.close()`, cancelled in `finally`. Apply the brain and start_all changes.
- [ ] **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit.** `git commit -m "fix(s2s): cap LLM turn time and stop dropping long speech"`

### Task 6: The stack notices a dead or wedged process

`start_all.sh` ends in a bare `wait`, so a dead s2s, tap or llama is never noticed and the wake daemon keeps saying ready. If an s2s handler wedges, its only pool unit goes "stuck" forever while `/v1/pool` still returns 200, and every reconnect gets `session_limit_reached`.

**Files:**
- Modify: `scripts/run_s2s.py:238` (`main()` starts a pool watchdog thread)
- Modify: `scripts/start_all.sh:228` (final wait), `:172-176` (s2s respawn loop)
- Modify: `familiar_brain/wake_daemon.py:112-117` (`_is_stack_live`), `:167-202` (`status`)
- Test: `tests/test_run_s2s.py`, `tests/test_start_all_logs.py`, `tests/test_wake_and_reachy_app.py`

**Interfaces:**
- Produces: `pool_is_wedged(pool_json: dict, max_drain_s: float) -> bool` in `scripts/run_s2s.py`, imported by `wake_daemon.py` through a small shared module `familiar_brain/stack_health.py` (put the function there and re-export it from `run_s2s.py`, so the brain package does not import from `scripts/`). Env var `S2S_STUCK_EXIT_S` (default 45). s2s exits with code 70 when wedged.

- [ ] **Step 1: Write the failing tests.** `test_pool_is_wedged_flags_stuck_and_long_drain`: `{"units":[{"state":"stuck"}]}` with 45 is True; `draining` with `draining_for_s` 60 is True; `draining` 3 is False; `idle` and `active` are False. `test_start_all_exits_when_child_dies`: stub s2s exits after 1 s and stub tap stays up; with respawn disabled (`S2S_RESPAWN=0`) `start_all.sh` exits non-zero within 5 s and the log names the dead child. `test_s2s_respawns_once`: with respawn on, the stub s2s is started twice. `test_stack_not_live_when_s2s_pool_stuck`: `/api/state` 200 and `/v1/pool` 200 with a stuck unit makes `_is_stack_live()` False.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.** Pool watchdog thread polls `/v1/pool` every 5 s and calls `os._exit(70)` when wedged. In `start_all.sh`, wrap the s2s launch in a loop that restarts it up to 5 times per 10 minutes and logs each restart; replace the final `wait` with `wait -n` over the long-lived PIDs, log which one exited, and `exit 1`. The wake daemon uses `pool_is_wedged` in `_is_stack_live`.
- [ ] **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit.** `git commit -m "fix(launch): detect and recover dead or wedged stack processes"`

### Task 7: Companion UI recovery controls work

Verified bugs that make the laptop UI look frozen or make recovery buttons do nothing.

**Files:**
- Modify: `familiar_brain/static/app.js:523` (`lastState` to `latestState`), `:653` (define `applyState(s)` that sets `latestState` and calls `renderStatus(s, false)`), and `apiJson` (5 s `AbortController` timeout so `pollInFlight` always clears)
- Modify: `familiar_brain/static/companion_plus.js` (stop the 750 ms boot-status poll and hide Speak Test / Reset Audio after the first 404, with the note "Only available from the robot UI on port 8043")
- Modify: `familiar_brain/app.py:880-886` (`api_mute_stop` calls `_publish_state()`), the error banner clears on the next good poll
- Modify: `familiar_brain/app.py` `_on_campaign_switched` (restart a session if one was active)
- Test: `tests/test_companion_ui.py`, `tests/test_campaign.py`

**Interfaces:**
- Produces: `POST /api/robot/reconnect` must reach the native app's `:8043` reconnect route when `state.robot.mode == "native"` (check what the route does today; if it only handles the official app, forward to `http://<robot>:8043/_familiar/reconnect`).

- [ ] **Step 1: Write the failing tests.** `test_app_js_has_no_undeclared_state_refs`: `lastState` not in app.js, and `function applyState` is defined. `test_companion_plus_handles_missing_familiar_routes`: companion_plus.js clears the boot-status interval on 404. `test_mute_stop_publishes_state`. `test_campaign_switch_keeps_session_active`: start a session, create and switch to campaign B, assert `/api/state` has a session whose file is under B's sessions dir. `test_robot_reconnect_route_reaches_native_app`: with mode native and a patched HTTP client, POST `/api/robot/reconnect` and assert one request to the robot's `:8043` reconnect path.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement** as listed under Files.
- [ ] **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit.** `git commit -m "fix(ui): working reconnect, live updates, and session survives campaign switch"`

### Task 8: Speaker ID does not leave Rocky ignoring the table

The tap keeps the last matched speaker (usually the DM) for up to 15 s because real clients never send `input_audio_buffer.commit`, so the gate ignores whoever speaks next. The tap also reads `data/speakers.json` while the brain enrolls into the active campaign's file, and `_tap_worker` dies silently on any exception.

**Files:**
- Modify: `familiar_brain/audio_tap.py` (`_run_match` streaming call uses `clear_on_miss=True`; `_tap_worker` loop catches, logs at WARNING and continues; read `session.audio.input.format.rate` from GA `session.update`)
- Modify: `familiar_brain/app.py` `_notify_tap_reload` (always send `store_path`; retry at startup until the tap answers)
- Modify: `config/brain.toml` (`speakers.max_age_s = 3.0`)
- Test: `tests/test_speaker_id.py`, `tests/test_ambient_audio.py`

**Interfaces:**
- Consumes: nothing new.

- [ ] **Step 1: Write the failing tests.** `test_tap_role_clears_on_streaming_miss`: registry stubbed to match DM twice then `(None, 0.3)`; feed 3 s of appends through `_inspect_client_text`; assert `current_state()["role"]` is None. `test_tap_worker_survives_exception`: first `_run_match` raises `ValueError`; assert the worker task is still running and a WARNING was logged. `test_tap_reads_ga_input_rate`: a GA `session.update` with rate 24000 sets `state["sr"]` to 24000. `test_notify_tap_reload_always_sends_store_path`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement** as listed.
- [ ] **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit.** `git commit -m "fix(tap): speaker label expires, worker survives errors, GA rate honoured"`

### Task 9: Official app path hardening (repo side)

The official app 1.0.1 recorded in `docs/smoke-test.md` has no send timeout on mic frames, so a dead network path freezes its record loop for up to 15 minutes, and a barge-in pauses the shared GStreamer pipeline that also carries the mic. Pollen main (1.2.0) adds a 5 s send timeout. The robot-side upgrade is a runbook step in Task 10; the repo side makes the mismatch visible and removes a needless double restart at launch.

**Files:**
- Modify: `scripts/robot_connect.py` (read the app version over `/rpc` or the daemon app listing; print a WARNING when older than the first release with the send timeout, and record it in the robot_connect log; skip `backend.config` when host and port are unchanged; drop `force=True` on `personalities.apply`)
- Modify: `docs/smoke-test.md:19-20` (expected app version)
- Test: `tests/test_robot_connect.py`

**Interfaces:**
- Produces: `parse_app_version(listing: dict) -> tuple[int, int, int] | None` and `MIN_APP_VERSION = (1, 2, 0)` in `robot_connect.py` (confirm the exact first version with `AUDIO_SEND_TIMEOUT_S` from Pollen's changelog before fixing the constant).

- [ ] **Step 1: Write the failing tests.** `test_warns_on_old_app_version` (1.0.1 logs a warning naming the minimum), `test_no_warning_on_current_version`, `test_configure_skips_backend_config_when_unchanged` (fake RPC records calls; `backend.config` not called when already pointed at the same host and port), `test_apply_is_not_forced`.
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit.** `git commit -m "fix(robot_connect): warn on old app, avoid double backend restart"`

### Task 10: Live verification on the laptop and robot (TJ, not subagents)

Run after Tasks 1 to 9 are merged to `main` and the laptop has pulled them. Each probe has a pass condition; record results in `docs/smoke-test.md`.

- [ ] **Step 1: Baseline.** `uv run pytest -q` on the laptop. Expected: no new failures versus the count recorded in Task 1.
- [ ] **Step 2: Confirm which robot app runs.** If it is the official app and the robot has internet, update it from the robot dashboard to the newest release, then rerun `launch_rocky.bat` and check the robot_connect log shows no version warning.
- [ ] **Step 3: Tap freeze probe.** With Rocky listening, in WSL run `kill -STOP $(pgrep -f familiar_brain.audio_tap); sleep 12; kill -CONT $(pgrep -f familiar_brain.audio_tap)`, then speak. Pass: transcription resumes within 5 s with no button press, and the run's logs show either nothing or a WARNING plus reconnect.
- [ ] **Step 4: Slow reply probe.** Ask Rocky a question while `/api/chat` is busy with a long silent-chat prompt (or with `FAMILIAR_LLM_DEADLINE_S=5` and a cold model). Pass: the next spoken turn is answered.
- [ ] **Step 5: Process death probe.** `pkill -f run_s2s.py`. Pass: `stack.log` names s2s, it restarts, and Rocky hears again within about 60 s (model reload time).
- [ ] **Step 6: Wi-Fi blip probe.** Toggle the laptop Wi-Fi off for 10 s and back on. Pass: Rocky hears again within 30 s without touching anything.
- [ ] **Step 7: Soak.** Leave Rocky in a normal two-hour conversation (or play audio of a session at it). Pass: no stall; if one happens, the per-run logs from Task 1 now say which layer failed, and that becomes the next fix.
