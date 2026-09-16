# Smoke test and probes for Plan 3

Run with the robot on, `./scripts/start_all.sh` up, and a session started. Record
every result in docs/versions.md under "Plan 3 probes". A probe that fails stops the
check at that point; do not work around it silently.

## Before you start

1. Firewall. The Hyper-V inbound rule for port 8765 did not exist on 2026-09-16
   (`Get-NetFirewallHyperVRule -Name WSL-s2s-8765` finds nothing). Create it from an
   administrator PowerShell with the command in docs/versions.md, "Firewall, one
   elevated step still required". Without it the robot cannot reach speech-to-speech.
2. Start the stack in WSL: `./scripts/start_all.sh`. Read the host LAN address from
   its "Point the robot at" line, from `./scripts/check_env.sh`, or from `ipconfig` on
   Windows. It was 10.0.0.95 on 2026-09-16 but moves with DHCP (docs/versions.md, "The
   host LAN address moves"), so read it fresh every time.
3. On the robot dashboard at http://10.0.0.100:8000, start
   `reachy_mini_conversation_app` (installed beside `rocky_mini` and
   `hand_tracker_v2`; no app was running on 2026-09-16). Note the app version it
   shows: `robot_profile/familiar/profile.md` is written for 1.0.1.
4. In the app's Settings view, Connection section, choose a local target of
   `<host LAN IP>:8765`, so the app connects to `ws://<host LAN IP>:8765/v1/realtime`.
5. Select the `familiar` profile. If it is not listed, add it as the app README's
   "Custom profiles" section describes: "Custom" with the name `familiar` and the body
   of `robot_profile/familiar/profile.md` as instructions, then "Manage tools" set to
   exactly the six `default_tools` in that file.
6. Start a brain session: `curl -X POST http://127.0.0.1:8090/api/session/start`.

The timing lines below are the JSON lines `familiar.timing` writes to data/brain.log,
one per request, each with a `"decision"` field.

## Probes (in order)

P0. Tool-result echoes. Say a line that is not addressed to Rocky. In data/brain.log
    count the timing lines it produced. One line (`"decision": "IGNORE"`) means the
    app does not return idle_do_nothing's result; two (IGNORE then ECHO) means it does
    and the empty response closed it. Either is fine. A third line, or speech, is a
    defect.

P1. Gesture plus speech in one response. "Rocky, what do you make of the smell?"
    Expect: head turns to you, an expression plays, then Rocky speaks. brain.log shows
    "gesture mood=" and "head tracking on". If the robot plays the emotion but never
    speaks, speech-to-speech dropped the text after a tool call: record it and stop.

P2. Tracking hands back. Stay quiet for follow_up_window_s + tracking_linger_s (8 + 4
    seconds in config/brain.toml today), then say any table line. brain.log shows
    "head tracking off". Say a line from the left of the robot: expect "glance left"
    and a head turn. The DoA payload shape the brain reads, `{"angle": <rad>,
    "speech_detected": <bool>}`, was confirmed against the live daemon on 2026-09-16
    (angle 2.39 rad, speech_detected false, nobody speaking). If the head does not
    move, GET http://10.0.0.100:8000/api/state/doa by hand while speaking from the left
    and record the payload: 0 is left, pi/2 front, pi right.

P3. Narration reactions. Read three lines as the Dungeon Master, at least fifteen
    seconds apart, none naming Rocky:
      a. "The priest pulls back his hood. It is the king, and he has been dead a year."
         Expect the surprised move; brain.log shows "reaction call answered 'alarmed'".
      b. "She folds the letter and does not read the rest aloud."
         Expect the sad move.
      c. "You walk for an hour along the river road."
         Expect nothing; brain.log shows the reaction call answered 'none'.
    Then read a and b again ten seconds apart: the second one must produce no
    reaction (reaction_gap_s). Record the `preface_ms` figure from the timing lines. If
    it is over 100 ms, or if c reacts, note it: D9's reversal applies.

P4. Sight. "Rocky, what do you see?" The brain does not log the tool calls it relays,
    so the camera call shows up when its frame comes back: expect a timing line with
    `"decision": "SIGHT"`, then a spoken description. Record how many SIGHT lines one
    question produced. data/llama.log must show the mmproj loaded and an image being
    encoded. Record VRAM from nvidia-smi with the frame in flight. If llama-server
    returns 4xx on the image request, record the body and stop: the reversal in D5
    applies.

P5. Dice. Put a d20 face up 30 to 50 cm in front of the camera, good light. Ask
    "Rocky, what did I roll?" ten times with different faces. Record hits. Below 7 of
    10, dice sight is dropped from v1 (decision D6) and the sight block in persona.py
    loses the words "read dice or".

P6. Thinking mode. "Rocky, think with me about the Drowned Pearl." Expect a longer
    reply ending in a question. "Rocky, thanks, that's all." Expect two-sentence replies
    again. /api/state shows mode.

## Hardware checklist (design section 9)

- mute cue visible (sleepy gesture on mute, welcoming on unmute)
- wake by name: "Rocky"
- dice roll spoken correctly (tool total matches speech)
- rule lookup
- follow-up window
- Speak Now from the phone
- head turns to the speaker when addressed (P1)
- glances during silence (P2)
- a twist and a sad line each earn a reaction, a travel line does not (P3)
- camera question answered locally (P4)
- mute closes the eyes (camera continuation while muted produces no speech)
- end session produces a summary file
