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

P0. Tool-result echoes. The brain sends idle_do_nothing only on a turn with nothing
    else to send, and the first turn after the brain starts carries "head tracking on"
    instead, whoever it was said to (unless the app's greeting already switched it on),
    so use two lines. With `tail -f data/brain.log` open in WSL, say a short line not
    addressed to Rocky ("pass the crisps") and let the log settle. Three to five seconds
    later say a second ("cheers, thanks"). Keep both under five words and away from
    dice, crits, laughter, anyone going down, and grief, all of which earn a reaction.
    Judge only the second line, from what brain.log adds after it. It must log no "head
    tracking" or "gesture" line; if it does, say another such line at once and judge
    that one.
    Then count its timing lines. One (`"decision": "IGNORE"`) means the app does not
    return idle_do_nothing's result; two (IGNORE then ECHO) means it does and the empty
    response closed it. Either is fine. A third line, or speech, is a defect.

P1. Gesture plus speech in one response. "Rocky, what do you make of the smell?"
    Expect: an expression plays, then Rocky speaks, the head following your face
    throughout. brain.log shows "gesture mood=" and no new "head tracking" line, since
    tracking has been on since P0. If the robot plays the emotion but never speaks,
    speech-to-speech dropped the text after a tool call: record it and stop.

P2. Tracking stays on, sleeps and wakes (deviation P14). Stay quiet for twenty
    seconds, well past the follow-up window, then move along the table while saying a
    line that does not name Rocky. The head keeps following your face and brain.log
    logs no "head tracking" line. Mute from the phone and say any line: brain.log logs
    "head tracking off" and the sleepy gesture plays after it. Unmute and say any line:
    "head tracking on", then the welcoming gesture, and the head follows faces again.

P3. Narration reactions. Read three lines as the Dungeon Master, at least fifteen
    seconds apart, none naming Rocky:
      a. "The priest pulls back his hood. It is the king, and he has been dead a year."
         Expect the surprised move; brain.log shows "reaction call answered 'alarmed'".
      b. "She folds the letter and does not read the rest aloud."
         Expect the sad move.
      c. "You walk for an hour along the river road."
         Expect nothing; brain.log shows the reaction call answered 'none'.
    Then read a and b again five seconds apart, well inside reaction_gap_s (10 s): the
    second one must produce no reaction. Record the `preface_ms` figures from the timing lines;
    on these lines they are the reaction call alone. If one is over 100 ms, or if c
    reacts, note it: D9's reversal applies.

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
- head follows faces whenever Rocky is awake, addressed or not (P1, P2)
- tracking off while muted, back on at unmute (P2)
- a twist and a sad line each earn a reaction, a travel line does not (P3)
- camera question answered locally (P4)
- mute closes the eyes (camera continuation while muted produces no speech)
- end session produces a summary file
