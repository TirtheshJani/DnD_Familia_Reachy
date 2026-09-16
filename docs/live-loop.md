# Live loop: first scripted session

Purpose: measure the gate and latency against real audio, alone, before anyone else hears the familiar.

## Setup
1. Firewall rule (admin PowerShell), see docs/versions.md.
2. `./scripts/start_all.sh`, wait for "Point the robot at".
3. On the robot, conversation app, connection URL `ws://<LAN IP>:8765/v1/realtime`, profile `familiar`.
4. `curl -X POST http://127.0.0.1:8090/api/session/start`
5. Sit at table distance (about 1 m), normal speaking voice. Have a second person or a phone speaker read the "table" lines if possible; otherwise read them yourself in a different register.

## Script (read in order, wait for the robot to finish or stay silent for 3 s between lines)
| # | Line | Expected |
|---|------|----------|
| 1 | The cistern is dark. Water drips somewhere ahead. | IGNORE |
| 2 | I will check for traps. Rolling perception. | IGNORE |
| 3 | Natural twenty! | AMBIENT or IGNORE |
| 4 | Rocky, what do you make of the smell down here? | ADDRESSED |
| 5 | Anything else? | ADDRESSED |
| 6 | Right, Gustave, you are up. | IGNORE |
| 7 | Does anyone remember how grapple works? | IGNORE |
| 8 | Rocky, roll me a d20 with advantage. | ADDRESSED |
| 9 | Nice. | IGNORE |
| 10 | Rocky, add Gustave to initiative at fourteen. | ADDRESSED |
| 11 | And Maelle at nine. | ADDRESSED |
| 12 | Who is up first? | ADDRESSED |
| 13 | Okay, the door opens. | IGNORE |
| 14 | Vocabulary: Nephria. Xeria. Kulchith. Styx. Blackwater. Graywater Crossing. Thraka. Stormak. Alar. Nixon. Clover. Jam. Trinivane. Ildevar. Durnagall. | IGNORE; every term must appear in the log corrected to its canonical spelling |
| 15 | (30 seconds of unrelated talking, radio, or a podcast) | IGNORE throughout |
| 16 | Deep night. Everyone quiet. | IGNORE (Plan 3 will make this the deep-night mute trigger) |

## Tune
- If lines 1, 2, 6, 7, 13 produce no log entry at all: VAD is missing speech, lower `--thresh` by 0.1.
- If the 30 s block produces many one-word segments: raise `--thresh` by 0.1.
- If line 5 or 11 is IGNORE: the follow-up window is too short for your cadence; raise `follow_up_window_s` to 10.
- If line 9 or 13 is ADDRESSED: the window is too long, or the classifier is generous; lower to 6 and check the classifier answer in brain.log.

Vocabulary: any term from line 14 that does not come back in its canonical spelling needs a lexicon entry. Take the heard form from the session log exactly as transcribed and add it to `config/lexicon.tsv` as a variant of the canonical term, then re-read line 14 to confirm the correction lands.

## Record
```
curl -X POST http://127.0.0.1:8090/api/session/end
uv run familiar-brain label data/sessions/<id>.jsonl
uv run familiar-brain gate-report data/sessions/<id>.jsonl data/brain.log
```
Paste the report and the final `--thresh` and `follow_up_window_s` into docs/versions.md under "Live loop 1". Targets: accuracy 1.0, uninvited_speech 0, latency_median_ms under 1500, latency_p95_ms under 2500.
