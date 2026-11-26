# Rocky audio stall: investigation findings (2026-11-26)

This is the evidence behind `2026-11-26-rocky-audio-stall-and-polish.md`. It was produced by reading the repo at `0a8c3eb`, the installed speech-to-speech 1.0.0 source, reachy_mini 1.11.0 and Pollen's reachy_mini_conversation_app (main, 1.2.0). Seven finders produced 70 hypotheses (61 after merging duplicates). The 19 that best fit the symptom were then checked by an independent skeptic told to refute them, with repros where cheap. Nothing here has been run on the real robot yet, and the 2026-11-25 run left empty logs, so none of these is proven to be the one TJ hit.

TJ's symptom: "it starts up and starts transcribing then just stops and doesn't work", with no recovery.

## Which robot app matters

The default launcher (`launch_rocky.bat` to `launch.ps1` to `start_all.sh`) never passes `--native-robot`, so it runs Pollen's official app through `robot_connect.py`. The native `reachy_familiar` app runs only when the wake daemon starts it. The ranking below depends on which one was running.

## Ranked causes that survived refutation

1. **Native app: mic thread dies silently after a send stall** (plausible, strong fit, reproduced against the real class). `audio_bridge.py:144-145` sets a 0.5 s socket timeout that also applies to the mic thread's `sendall` (`:131-132`). A send stall over 0.5 s raises, `_mic_loop` logs at DEBUG and exits (`:423-425`), and `_run_session` keeps answering pings with `connected=True`. Measured on Linux: a timed-out send writes no partial frame, so framing stays valid and nothing ever reconnects. A stall of a few seconds anywhere downstream (s2s not reading, so the tap blocks in `drain()` at `audio_tap.py:430-432`; WSL2 or Wi-Fi hiccup) is enough. Plan Task 2.
2. **Both paths, mostly official: an unanswered robot tool call blocks every later reply** (plausible, verified in code and by running it). The brain sends a gesture or `idle_do_nothing` tool call on nearly every turn (`app.py:774`). s2s refuses all later LLM requests while any call lacks an output, with no timeout and no eviction (`chat.py:142-215`, `base_openai_compatible_language_model.py:943,1069-1076`). The official app drops its tool result if `response.done` takes over 30 s (`huggingface_realtime.py:69,782-794`). One slow reply then silences Rocky until reconnect. Transcription still shows, replies stop. Plan Task 4.
3. **Server: one brain request can hold the only LLM thread for about 60 s** (plausible). 20 s per read, two SDK retries, and the brain sends no headers until llama's first token. Plan Task 5.
4. **Server: a wedged s2s handler leaves the single pool unit stuck forever** while `/v1/pool` returns 200, so every reconnect is rejected with `session_limit_reached`. Rare by itself, but it turns any wedge into permanent silence. Plan Task 6.
5. **Native app: dead mic capture is never restarted** (plausible). Plan Task 3.
6. **Official app 1.0.1/1.1.x: no send timeout on mic frames** (plausible, partial fit). Only permanent when the network path is truly dead (laptop sleep, Wi-Fi reset, DHCP move); then it freezes for up to about 15 minutes. Fixed upstream with a 5 s timeout. Plan Tasks 9 and 10.
7. **Official app: every barge-in pauses the shared GStreamer pipeline that also carries the mic** (plausible, unproven on hardware). Upgrading the app or SDK is the practical fix. Plan Task 10.
8. **`--max_speech_ms 25000` discards any segment over 25 s with no transcript** (plausible). Plan Task 5.

## Refuted or demoted

The skeptics refuted these as causes of the symptom: proxy liveness (uvicorn's 20 s ping already covers it), compaction overflow (s2s evicts synchronously), blocking reopen-gate waits (pre-empted, about 2 s jitter at most), the s2s receive loop coupling, "Mute & Stop Logging" and "deep night" (both visible states, and transcription continues). The native per-byte masking and the tap's frame codec were measured and ruled out.

## Evidence gap

`start_all.sh` truncates every log on each launch and the wake daemon's output is lost under `pythonw.exe`. Plan Task 1 fixes this first.

## Polish backlog (verified bugs)

High: laptop Reconnect button uses an undefined `lastState`; tap keeps a stale DM label so Rocky ignores the next speaker; unanswered tool call (above); native mic death (above). Medium: SSE listener calls undefined `applyState`; campaign switch ends the session and starts none; Speak Test / Reset Audio call routes that only exist on the robot; tap and brain read different `speakers.json`; dead children go unnoticed. Low, deferred unless time allows: `/api/roll` and `/api/behavior` 500 on bad input, stale error banner, wake daemon uninstall leaves the launcher, native `stop_emotion` needs a move UUID, native emotions dropped while another move runs, ambient mood never goes stale, behaviour settings save edge cases, speaker registry drops profiles on a corrupt store.

## Pollen parity summary

On the official-app path Rocky already gets Pollen's layered motion, idle breathing, speech wobble, antenna freeze while listening, face tracking, camera and the emotions library for free. The native app lacks layered motion, breathing and antenna freeze, and porting them is a large job. For "use the robot's full capacity" by tonight, the official app on its newest release is the better path; the native app is worth keeping only as a fallback.

## Test baseline

In a clean cloud copy without torch: 766 passed, 25 failed, 10 skipped. All 25 failures are missing gitignored local data (`data/notes/10-brokk-trenevain.md`, `config/persona.yaml`, `data/sessions/20260916-*`), not code.
