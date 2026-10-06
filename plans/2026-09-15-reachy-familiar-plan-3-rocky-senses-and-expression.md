# Reachy Familiar, Plan 3: Rocky, Expression, Attention and Sight

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the familiar to Rocky and give it a body language: it gestures with every reply, reacts to what the Dungeon Master narrates and to the table's outbursts (a shocking twist earns a surprised look, a sorrowful line a sad one), turns to whoever is talking to it, can look at the room and the dice through its camera, and can switch into a longer thinking mode when asked.

**Architecture:** Nothing new runs on the robot. The brain already sits between speech-to-speech and llama-server, and the only channel it has to the motors is the Responses stream it returns, so every gesture, glance and tracking toggle rides on a turn as a `function_call` item for a tool the conversation app owns (`play_emotion`, `move_head`, `head_tracking`). Those tools are marked `needs_response = False` in the app, so a gesture never costs a second generation. The mood behind a gesture is chosen by the brain: keyword rules first, then a one-token call on the classify slot, never by the generating model. Sight uses the app's own `camera` tool: the app attaches the frame as an `input_image` item, speech-to-speech forwards it, and the brain lets that continuation through the gate to llama-server running Gemma 4 with its multimodal projector. Sound direction is read from the robot daemon over REST to aim glances.

**Tech Stack:** Existing Plan 1 and 2 stack. No new Python dependencies. llama-server gains the Gemma 4 `mmproj` file (auto-downloaded with `-hf`, see Task 12).

**Design:** `docs/superpowers/specs/2026-09-10-reachy-familiar-design.md` sections 4.7 (vision and liveliness), 5 (decision effects) and 7 (persona). Read `docs/deviations.md` first: Plan 1 and 2 were both modified during execution and the snippets below are written against the code as it stands on 2026-09-16.

**Ground rule for this plan:** before editing any existing file, read it as it is now. Snippets name functions, not line numbers. If a snippet conflicts with the current code, the current code wins on structure and this plan wins on behaviour. Write the behaviour test first either way. British spelling in prose, no em or en dashes anywhere.

**Status (2026-09-16):** TJ confirmed decisions D1 to D8 below and added the narration reactions (D9). Deep night is out of this plan until TJ has talked to the Dungeon Master (see "Deferred"). The live check with the robot is deferred to a later session; Tasks 1 to 12 are all doable without the robot, and everything that needs it is collected under "Deferred" at the end.

---

## Decisions this plan assumes

These came out of grilling the request against the design document, the installed conversation app 1.0.1 source, speech-to-speech 1.0.0, the reachy_mini SDK, and the earlier `rocky_mini` project. TJ confirmed them on 2026-09-16. Each has a reversal.

| # | Decision | Reasoning | Reversal |
|---|----------|-----------|----------|
| D1 | The name is Rocky and only the name carries over from `rocky_mini`. No Eridian voice, chords, or memory. | The design leaves the creature concept to Session Zero; the familiar is a different character from the Project Hail Mary Rocky. | If TJ wants the Eridian character, Plan 4 ports `rocky_mini/audio/dsp.py` ring-mod into a TTS post-filter. Nothing in this plan blocks it. |
| D2 | Expression is chosen by the brain, not the model. One gesture per turn, picked by keyword rules and a one-token mood call on the classify slot, rate limited to one every 6 s. | Gemma 4 through llama-server has to be told to call a tool and then reliably does or does not (deviation E3). A deterministic gesture is testable and never silent. The robot's own idle policy (60 percent still, 16 percent emotion, 8 percent head move) already covers liveliness between turns. | If the mood call costs more than 100 ms live, drop it and keep the keyword rules. |
| D3 | Persona gestures name the app's emotion intents (`happy`, `surprised`, `thinking`, `success`, `attentive`, `sad`, `scared`, `sleepy`, `welcoming`), not recorded move ids. | `play_emotion` in app 1.0.1 resolves an intent to a curated move. A raw id such as `proud1`, which `persona.example.yaml` uses today, is not in its allow-list and silently becomes a random curated move. | If the robot runs an older app that lacks intents, map back to the move ids listed in deviation C7. |
| D4 | Attention follows design 4.7: daemon head tracking is on while Rocky is addressed and for a few seconds after the follow-up window, off otherwise. Off-tracking glances use the daemon's sound direction reading. The brain never writes to the daemon REST API; it only reads `/api/state/doa`. | The app's `MovementManager` owns the head and already pauses tracking while speaking. Writing to the daemon from a second process would fight it (the footgun `rocky_mini` learnt the hard way). Turn-driven toggles need no background thread. | If the attend feel is too sparse, Plan 4 adds a brain-side timer that emits a glance on the next turn. |
| D5 | Sight goes through the existing chain: app `camera` tool, speech-to-speech `input_image`, brain, llama-server with the Gemma 4 mmproj. No SmolVLM2. Frames are never logged. | Deviation D confirmed the mmproj exists; versions.md measured 740 MiB of headroom, which a second vision model does not fit in. | If the mmproj does not fit or `/v1/responses` rejects images (probe P3), fall back to the 0.6B TTS before anything else, as the design says. |
| D6 | Dice sight is a measured probe, not a detector. Rocky is asked "what did I roll" ten times at 30 to 50 cm; below 7 of 10 it is dropped from v1 and Rocky reacts to announced rolls instead. | A trained dice detector is a project of its own. Reading pips with a general VLM at table distance is plausible only for large dice up close, so measure before building. TJ accepted the probe without asking for a trained model. | If TJ wants dice sight regardless of the probe, Plan 4 scopes a YOLO dice model on the Zephyrus. |
| D7 | No face identification. Design non-goal 2 stands: Rocky tracks and describes people, never names them. "Sees me" means it looks at and describes whoever is there. | Consent at a table of other people, and the design's explicit exclusion. TJ will ask the players for consent at Session Zero; until that is recorded, nothing in the code identifies anyone. | Session Zero consent decision, not a code one. If the players consent, Plan 4 scopes it; this plan leaves no hook for it on purpose. |
| D8 | "Thought partner" is a reply mode. Phrases open it ("think with me", "help me think", "talk it through"), phrases close it ("that's all", "quick answer"), it times out after 5 minutes without an addressed turn, and the companion UI can set it through `POST /api/mode`. In thinking mode the fixed length ceiling becomes six sentences ending in a question or a suggestion. | A mode is one line in the fixed blocks and one state field. It keeps the two-sentence ceiling as the default a table needs. | If phrase triggers misfire at the table, keep only the API and a Speak Now style button. |
| D9 | Rocky reacts to narration. On lines not addressed to it (IGNORE and AMBIENT alike), the brain picks a reaction from a short list (`amused`, `alarmed`, `sad`, `scared`) by keyword rules first and, for lines of five words or more with no rule hit, by a one-token reaction call on the classify slot that may also answer `none`. Reactions are rate limited separately (one every 15 s by default) on top of the 6 s gesture gap, and never fire while muted. | The gate deliberately classes Dungeon Master narration as IGNORE (deviation E8), so a reaction cannot hang off the AMBIENT decision alone. The reaction call is the same 59 ms shape as classify and runs only on turns that produce no speech, so it never delays a reply. A separate, longer gap keeps Rocky from twitching at every line of a long description. The brain cannot tell who is speaking, so "the Dungeon Master narrates" means "any line not spoken to Rocky". | If reactions fire on ordinary table logistics at the live check, raise the word threshold or the gap, or restrict the call to lines the gate classed AMBIENT. If the second call per line shows up in the gate's timing, fold the reaction into the classify prompt in Plan 4. |

---

## What the evidence established (read before coding)

- **The app never asks the model on idle time.** `ConversationHandler.send_idle_signal` runs a locally chosen idle tool; nothing reaches the brain. Between turns the robot breathes and occasionally plays a random curated emotion or head move on its own. `dance` is not in our profile, so the weights renormalise over the rest.
- **Gesture tools do not trigger a second generation.** `play_emotion`, `move_head`, `head_tracking`, `stop_emotion`, `sweep_look` set `needs_response = False`. The app sends their `function_call_output` back into the conversation but does not request a response. `idle_do_nothing` and `camera` keep the default `needs_response = True`, so the app does request a response after them. Whether that request actually reaches the brain for `idle_do_nothing` is probe P0 in the deferred live check; the brain answers such an echo with an empty response either way, which cannot loop.
- **The gate classes narration as IGNORE.** `CLASSIFY_INSTRUCTION` in `llm.py` lists Dungeon Master narration, rules questions and dice announcements as IGNORE and reserves AMBIENT for outbursts (deviation E8). That is right for speech and is not changed here. Reactions therefore run on both non-addressed decisions.
- **The fake LLM answers "IGNORE" when its queue is empty** (`tests/fakes/llm_server.py`). A reaction call that finds no queued reply gets "IGNORE", which is not a reaction mood and yields no gesture, so existing app tests are not disturbed by the extra call.
- **Camera frames reach the brain.** `huggingface_realtime.py` sends the `camera` result as a `function_call_output` (with `image_attached: true`) followed by a user message whose only content part is `{"type": "input_image", "image_url": "data:image/jpeg;base64,..."}`. speech-to-speech `Chat` accepts `input_image` and serialises it for the Responses API unchanged, then strips images from history after the reply.
- **Head tracking is daemon-side.** `MovementManager.set_head_tracking(True)` calls `start_head_tracking(weight=1.0)`; while speaking it drops the weight to 0 once a face is locked and restores it afterwards. The daemon endpoints `/api/media/tracking/{enable,disable,face}` exist (deviation D) but this plan does not call them.
- **Sound direction is one GET.** `GET /api/state/doa` on the daemon returns `{"angle": <rad>, "speech_detected": <bool>}` or `null`. 0 is left, pi/2 front, pi right.
- **`move_head` is invisible while tracking is on** (the daemon owns the head at weight 1), which is why glances only happen while tracking is off.
- **Profiles changed shape.** App 1.0.1 reads `profile.md` with a `+++` TOML header and `default_tools`; the repo's `robot_profile/familiar/` still has the older `instructions.md` and `tools.txt`. Task 12 adds `profile.md`.
- **The robot was unreachable from this session** (no answer from 10.0.0.100:8000), so nothing here was verified against the live daemon. Every live check is deferred.

---

## File structure

```
familiar_brain/
  responses_api.py   modify: call_item, item_events, tool_call_events, empty_response_events, idle_items, continuation_kind
  motion.py          create: moods, table and addressed rules, reaction list, Choreographer
  robot.py           create: RobotDaemon (DoA read only), direction_from_angle
  attention.py       create: Attention (tracking toggles and glances per turn)
  modes.py           create: thinking mode phrase rules
  persona.py         modify: mode-aware length block, sight block
  llm.py             modify: LLMClient.mood, MOOD_INSTRUCTION, REACTION_INSTRUCTION
  state.py           modify: pending cue, tracking, glance stamp, mode
  config.py          modify: [robot] and [motion] sections, defaulted
  app.py             modify: prefaces, continuation handling, reactions, /api/mode, /api/state fields
config/
  brain.toml         modify: [robot], [motion]
  persona.example.yaml  modify: Rocky, intent gestures
  lexicon.tsv        modify: Rocky variants
robot_profile/familiar/profile.md  create
scripts/start_all.sh  modify: explicit mmproj
docs/
  live-loop.md       modify: Rocky in the script
  smoke-test.md      create: probes P0 to P6 and the hardware checklist, for the deferred live check
  deviations.md      modify: Plan 3 section
tests/
  test_responses_api.py  modify
  test_motion.py  test_robot.py  test_attention.py  test_modes.py  test_state.py  create
  test_persona.py  test_llm.py  test_config.py  test_lexicon.py  test_app_integration.py  modify
  test_app_expression.py  test_app_sight.py  create
```

Each task ends green on the whole suite (`uv run pytest -q`), not only on its own file.

---

## Part A: Rocky

### Task 1: Rename the persona and teach the lexicon the name

**Files:**
- Modify: `config/persona.example.yaml`, `config/persona.yaml` (local, gitignored), `config/lexicon.tsv`, `docs/live-loop.md`
- Test: `tests/test_persona.py`, `tests/test_lexicon.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_persona.py`:

```python
def test_shipped_example_persona_is_rocky_with_intent_gestures():
    """The example file is what a fresh checkout copies to persona.yaml, so it has to
    load, be named Rocky, and use the app's emotion intents rather than move ids
    (app 1.0.1 play_emotion silently randomises an unknown id such as proud1)."""
    per = load_persona(Path(__file__).parent.parent / "config" / "persona.example.yaml")
    assert per.name == "Rocky" and "rocky" in per.aliases
    intents = {"attentive", "happy", "surprised", "thinking", "success", "sad", "scared", "sleepy", "welcoming"}
    for mood, (tool, arg) in per.gestures.items():
        assert tool == "play_emotion", mood
        assert arg in intents, f"{mood} uses {arg!r}, not an app emotion intent"
    for mood in ("attentive", "amused", "alarmed", "thoughtful", "proud", "sad", "scared", "asleep", "awake"):
        assert mood in per.gestures, mood
```

Append to `tests/test_lexicon.py` (read the file first for the constructor it uses; the shipped `Lexicon.load` takes the TSV path):

```python
def test_shipped_lexicon_corrects_heard_forms_of_rocky():
    from pathlib import Path
    from familiar_brain.lexicon import Lexicon
    lex = Lexicon.load(Path(__file__).parent.parent / "config" / "lexicon.tsv")
    assert lex.correct("Rockie, roll a d20") == "Rocky, roll a d20"
    assert lex.correct("hey Rockey") == "hey Rocky"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_persona.py tests/test_lexicon.py -v`
Expected: both new tests FAIL (name is Understudy; Rockie is not corrected).

- [ ] **Step 3: Rewrite `config/persona.example.yaml`**

```yaml
name: Rocky
aliases: [rocky]
tts_speaker: Aiden
manner: |
  Attentive and dry. Curious about mortals and their habits.
  Never louder than the table. Answers in one or two sentences.
# Gestures name the conversation app's emotion intents (play_emotion.py, app 1.0.1),
# which the app resolves to a curated recorded move. Do not use raw move ids here:
# an id the app does not allow-list is replaced by a random curated emotion.
# amused, alarmed, sad and scared are also the reactions Rocky may show to narration.
gestures:
  attentive: play_emotion:attentive
  amused: play_emotion:happy
  alarmed: play_emotion:surprised
  thoughtful: play_emotion:thinking
  proud: play_emotion:success
  sad: play_emotion:sad
  scared: play_emotion:scared
  asleep: play_emotion:sleepy
  awake: play_emotion:welcoming
```

Copy the same content over `config/persona.yaml` (it is gitignored; the live brain reads it). The `asleep` and `awake` moods are the mute and unmute cues (Task 11); they are not a deep night feature.

- [ ] **Step 4: Add the name to `config/lexicon.tsv`**

Append one line (tab separated):

```
Rocky	Rockie|Rockey|Roki|Rocki
```

Do not add `rock` or `rocket` as variants or aliases: "rock" is ordinary D&D talk ("throw a rock") and a word-boundary alias on it would open the gate for the whole table.

- [ ] **Step 5: Update `docs/live-loop.md`**

Replace every `Understudy,` in the script table with `Rocky,` and `Little one,` in line 8 with `Rocky,`. Leave the tuning and record sections alone.

- [ ] **Step 6: Run the suite**

Run: `uv run pytest -q`
Expected: all green. Existing tests keep their inline `Understudy` fixtures; the name is a parameter everywhere and those tests are about behaviour, not the shipped file.

- [ ] **Step 7: Commit**

```bash
git add config/persona.example.yaml config/lexicon.tsv docs/live-loop.md tests/test_persona.py tests/test_lexicon.py
git commit -m "feat(persona): the familiar is Rocky; gestures use app emotion intents"
```

---

## Part B: Expression

### Task 2: Generalise the tool-call response builders

**Files:**
- Modify: `familiar_brain/responses_api.py`
- Test: `tests/test_responses_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_responses_api.py`:

```python
from familiar_brain.responses_api import call_item, item_events, tool_call_events, empty_response_events, idle_items


def test_call_item_is_a_completed_function_call_with_json_arguments():
    item = call_item("play_emotion", {"emotion": "happy"})
    assert item["type"] == "function_call" and item["status"] == "completed"
    assert item["name"] == "play_emotion" and json.loads(item["arguments"]) == {"emotion": "happy"}
    assert item["call_id"].startswith("call_") and item["id"].startswith("fc_")
    assert call_item("idle_do_nothing")["arguments"] == "{}"


def test_idle_items_is_one_idle_call():
    (item,) = idle_items()
    assert item["name"] == "idle_do_nothing" and item["arguments"] == "{}"


def test_item_events_number_items_from_the_start_index():
    a, b = call_item("head_tracking", {"enabled": True}), call_item("play_emotion", {"emotion": "attentive"})
    evs = item_events([a, b], start_index=3)
    assert [e["type"] for e in evs] == [
        "response.output_item.added", "response.function_call_arguments.delta",
        "response.function_call_arguments.done", "response.output_item.done"] * 2
    assert [e["output_index"] for e in evs] == [3, 3, 3, 3, 4, 4, 4, 4]
    assert evs[0]["item"]["arguments"] == "" and evs[3]["item"] is a
    assert evs[1]["item_id"] == a["id"] and evs[1]["delta"] == a["arguments"]


def test_tool_call_events_wrap_items_in_a_response_envelope():
    items = [call_item("play_emotion", {"emotion": "happy"})]
    evs = tool_call_events(items, response_id="r9")
    assert evs[0]["type"] == "response.created" and evs[0]["response"]["id"] == "r9"
    assert evs[-1]["type"] == "response.completed" and evs[-1]["response"]["output"] == items


def test_empty_response_events_carry_no_items():
    evs = empty_response_events(response_id="r0")
    assert [e["type"] for e in evs] == ["response.created", "response.completed"]
    assert evs[-1]["response"]["output"] == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_responses_api.py -v`
Expected: ImportError on `call_item`.

- [ ] **Step 3: Implement in `familiar_brain/responses_api.py`**

Replace `_idle_item` and `idle_response_events` with the following (keep `IDLE_TOOL`, `latest_user_text`, `function_call_outputs`, `wants_stream`, `sse`, `response_json`, `idle_response_json`, `with_instructions_and_tools` as they are; `idle_response_json` keeps calling `_idle_item()`):

```python
def call_item(name: str, args: dict | None = None) -> dict:
    """A completed function_call item for a tool the robot owns.

    call_id must be client-supplied and prefixed 'call_': speech_to_speech's
    Chat.add_item rejects a function_call without one (docs/protocol-notes.md 3).
    """
    return {"id": f"fc_{uuid.uuid4().hex[:8]}", "type": "function_call", "status": "completed",
            "call_id": f"call_{uuid.uuid4().hex[:12]}", "name": name,
            "arguments": json.dumps(args or {}, separators=(",", ":"))}


def _idle_item() -> dict:
    return call_item(IDLE_TOOL)


def idle_items() -> list[dict]:
    """The one-item preface for a turn that has nothing else to send."""
    return [_idle_item()]


def item_events(items: list[dict], *, start_index: int = 0) -> list[dict]:
    """The added, delta, done, done quartet for each item, without the envelope.

    Used on its own to preface an upstream text stream with gestures: speech-to-speech
    consumes only output_item.done, output_text.delta and completed, so the preface can
    precede the upstream response.created without confusing it.
    """
    out: list[dict] = []
    for i, item in enumerate(items, start=start_index):
        out += [
            {"type": "response.output_item.added", "output_index": i,
             "item": {**item, "status": "in_progress", "arguments": ""}},
            {"type": "response.function_call_arguments.delta", "output_index": i,
             "item_id": item["id"], "delta": item["arguments"]},
            {"type": "response.function_call_arguments.done", "output_index": i,
             "item_id": item["id"], "arguments": item["arguments"]},
            {"type": "response.output_item.done", "output_index": i, "item": item},
        ]
    return out


def tool_call_events(items: list[dict], *, response_id: str | None = None) -> list[dict]:
    """A complete streamed response whose only output is the given tool calls."""
    rid = response_id or f"resp_{uuid.uuid4().hex[:12]}"
    return ([{"type": "response.created", "response": {"id": rid, "status": "in_progress", "output": []}}]
            + item_events(items)
            + [{"type": "response.completed", "response": {"id": rid, "status": "completed", "output": list(items)}}])


def idle_response_events(*, response_id: str | None = None) -> list[dict]:
    return tool_call_events(idle_items(), response_id=response_id)


def empty_response_events(*, response_id: str | None = None) -> list[dict]:
    """A completed response with nothing in it. The answer to a tool-result echo the
    client sends after a robot-side tool that needed no reply; it cannot start a loop
    because it carries no tool call for the client to execute."""
    return tool_call_events([], response_id=response_id)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_responses_api.py -v`
Expected: all PASS, including the four pre-existing idle tests.

- [ ] **Step 5: Commit**

```bash
git add familiar_brain/responses_api.py tests/test_responses_api.py
git commit -m "feat(responses): general tool-call item and event builders"
```

---

### Task 3: motion.py, the choreographer

**Files:**
- Create: `familiar_brain/motion.py`
- Test: `tests/test_motion.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_motion.py`:

```python
import json
from pathlib import Path

from familiar_brain.motion import (
    REACTION_MOODS, Choreographer, table_mood, addressed_mood, tracking_item, glance_item,
)
from familiar_brain.persona import load_persona

PERSONA = """
name: Rocky
aliases: [rocky]
tts_speaker: Aiden
manner: dry
gestures:
  attentive: play_emotion:attentive
  amused: play_emotion:happy
  alarmed: play_emotion:surprised
  thoughtful: play_emotion:thinking
  sad: play_emotion:sad
  scared: play_emotion:scared
  asleep: play_emotion:sleepy
  awake: play_emotion:welcoming
"""


def persona(tmp_path: Path):
    p = tmp_path / "p.yaml"; p.write_text(PERSONA)
    return load_persona(p)


def test_table_mood_rules():
    assert table_mood("Natural twenty!") == "amused"
    assert table_mood("oh no, nat 1") == "alarmed"
    assert table_mood("He went down! Get him up!") == "scared"
    assert table_mood("hahaha that is hilarious") == "amused"
    assert table_mood("They hold a small funeral at dawn.") == "sad"
    assert table_mood("the door creaks") is None


def test_reaction_moods_are_the_four_narration_reactions():
    assert REACTION_MOODS == ("amused", "alarmed", "sad", "scared")


def test_addressed_mood_rules_default_to_attentive():
    assert addressed_mood("Rocky, roll me a d20") == "attentive"
    assert addressed_mood("Rocky, what do you think of him?") == "thoughtful"
    assert addressed_mood("good boy Rocky") == "proud"
    assert addressed_mood("Rocky, hello") == "attentive"


def test_tracking_and_glance_items():
    t = tracking_item(True)
    assert t["name"] == "head_tracking" and json.loads(t["arguments"]) == {"enabled": True}
    g = glance_item("left")
    assert g["name"] == "move_head" and json.loads(g["arguments"]) == {"direction": "left"}


def test_choreographer_uses_classifier_then_rules_and_rate_limits(tmp_path: Path):
    now = {"t": 100.0}
    calls = []
    def classify(text):
        calls.append(text); return "thoughtful"
    c = Choreographer(persona(tmp_path), min_gap_s=6.0, reaction_gap_s=0.0,
                      clock=lambda: now["t"], mood_classifier=classify)
    items = c.for_addressed("Rocky, roll a d20", now["t"])
    assert [json.loads(i["arguments"])["emotion"] for i in items] == ["thinking"]
    assert calls == ["Rocky, roll a d20"]
    now["t"] += 2
    assert c.for_addressed("Rocky, again", now["t"]) == []          # inside the gap
    now["t"] += 5
    assert len(c.for_addressed("Rocky, again", now["t"])) == 1       # gap elapsed


def test_choreographer_falls_back_when_classifier_fails_or_answers_nonsense(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                      mood_classifier=lambda t: (_ for _ in ()).throw(RuntimeError("down")))
    assert json.loads(c.for_addressed("Rocky, what do you think", 0.0)[0]["arguments"]) == {"emotion": "thinking"}
    c2 = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                       mood_classifier=lambda t: "banana")
    assert json.loads(c2.for_addressed("Rocky, hello", 0.0)[0]["arguments"]) == {"emotion": "attentive"}


def test_table_reaction_uses_rules_first_and_cues_ignore_the_gap(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=60.0, reaction_gap_s=0.0, clock=lambda: 0.0)
    assert c.for_table("the door creaks", 0.0) == []
    assert json.loads(c.for_table("natural 20", 0.0)[0]["arguments"]) == {"emotion": "happy"}
    assert c.for_table("natural 20", 1.0) == []                       # rate limited
    assert json.loads(c.cue("asleep", 1.0)[0]["arguments"]) == {"emotion": "sleepy"}   # cues always fire


def test_table_reaction_asks_the_classifier_for_narration_of_five_words_or_more(tmp_path: Path):
    asked = []
    def react(text):
        asked.append(text); return "sad"
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=15.0, clock=lambda: 0.0,
                      reaction_classifier=react)
    line = "The letter is from her brother, dead these ten years."
    assert json.loads(c.for_table(line, 0.0)[0]["arguments"]) == {"emotion": "sad"}
    assert asked == [line]
    assert c.for_table("A shape moves behind the altar and laughs.", 5.0) == []   # inside reaction gap, no call
    assert asked == [line]
    assert c.for_table("short line here", 20.0) == [] and asked == [line]        # under five words, no call


def test_table_reaction_treats_none_nonsense_and_errors_as_no_gesture(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                      reaction_classifier=lambda t: "none")
    assert c.for_table("You walk for an hour along the road.", 0.0) == []
    c2 = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                       reaction_classifier=lambda t: "thoughtful")               # not a reaction mood
    assert c2.for_table("You walk for an hour along the road.", 0.0) == []
    c3 = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0,
                       reaction_classifier=lambda t: (_ for _ in ()).throw(RuntimeError("down")))
    assert c3.for_table("You walk for an hour along the road.", 0.0) == []


def test_mood_without_a_gesture_yields_nothing(tmp_path: Path):
    c = Choreographer(persona(tmp_path), min_gap_s=0.0, reaction_gap_s=0.0, clock=lambda: 0.0)
    assert c.cue("proud", 0.0) == []   # persona above defines no proud gesture
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_motion.py -v`
Expected: ModuleNotFoundError `familiar_brain.motion`.

- [ ] **Step 3: Create `familiar_brain/motion.py`**

```python
"""Expression: which robot-side gesture goes out with a turn.

The brain cannot move the robot between turns. Its only channel to the motors is the
Responses stream speech-to-speech relays, so every gesture rides on a turn as a
function_call item for a tool the conversation app owns: play_emotion, move_head,
head_tracking. Those three set needs_response=False in the app, so a gesture never
costs a second generation (docs/protocol-notes.md section 4, plan 3 evidence notes).

Two kinds of gesture. An ADDRESSED turn carries the mood Rocky answers in. A turn not
addressed to Rocky (IGNORE or AMBIENT) may carry a reaction to what was said: the
Dungeon Master's twist, a death, a joke. The gate classes narration as IGNORE on
purpose (deviation E8), so reactions cannot hang off AMBIENT alone.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable

from familiar_brain.persona import Persona
from familiar_brain.responses_api import call_item

log = logging.getLogger("familiar.motion")

MoodClassifier = Callable[[str], str | None]

# The reactions Rocky may show to a line not spoken to it. The reaction call may also
# answer "none"; anything outside this tuple is no gesture.
REACTION_MOODS: tuple[str, ...] = ("amused", "alarmed", "sad", "scared")

# Lines shorter than this never reach the reaction call: "pass the chips" is not
# narration, and the keyword rules already cover short outbursts.
MIN_REACTION_WORDS = 5

# Design section 5: a non-addressed line may emit one gesture chosen by a small keyword
# rule set. These run before the reaction call and cost nothing.
TABLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(nat(ural)? ?(twenty|20)|crit(ical)?( hit)?)\b", re.I), "amused"),
    (re.compile(r"\b(nat(ural)? ?(one|1)|fumble)\b", re.I), "alarmed"),
    (re.compile(r"\b(went down|(he|she|they)('s| is| are) (down|dead|dying)|get (him|her|them) up|unconscious)\b", re.I), "scared"),
    (re.compile(r"\b(ha ?ha|hahaha|hilarious)\b", re.I), "amused"),
    (re.compile(r"\b(funeral|farewell|grief|mourn(s|ing)?|weep(s|ing)?|in tears|passed away)\b", re.I), "sad"),
)

# Fallback for ADDRESSED turns when the one-token mood call is unavailable or wrong.
ADDRESSED_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(good (boy|girl|job)|well done|clever)\b", re.I), "proud"),
    (re.compile(r"\b(what do you (think|make|reckon)|thoughts|opinion)\b", re.I), "thoughtful"),
    (re.compile(r"\broll\b", re.I), "attentive"),
)
DEFAULT_ADDRESSED_MOOD = "attentive"


def table_mood(text: str) -> str | None:
    for pattern, mood in TABLE_RULES:
        if pattern.search(text):
            return mood
    return None


def addressed_mood(text: str) -> str:
    for pattern, mood in ADDRESSED_RULES:
        if pattern.search(text):
            return mood
    return DEFAULT_ADDRESSED_MOOD


def tracking_item(enabled: bool) -> dict:
    return call_item("head_tracking", {"enabled": bool(enabled)})


def glance_item(direction: str) -> dict:
    return call_item("move_head", {"direction": direction})


class Choreographer:
    """Turns a mood into at most one gesture item per turn, rate limited.

    Emotions queue on the robot and each plays for a few seconds; a rapid exchange
    without a gap would pile them up and Rocky would still be emoting a minute later.
    Reactions to the table have a second, longer gap of their own so a long description
    earns one look, not one per sentence. Cues (mute, unmute) bypass both gaps because
    they are deliberate state changes.
    """

    def __init__(self, persona: Persona, *, min_gap_s: float, reaction_gap_s: float,
                 clock: Callable[[], float], mood_classifier: MoodClassifier | None = None,
                 reaction_classifier: MoodClassifier | None = None):
        self.persona = persona
        self.min_gap_s = min_gap_s
        self.reaction_gap_s = reaction_gap_s
        self.clock = clock
        self.mood_classifier = mood_classifier
        self.reaction_classifier = reaction_classifier
        self.last_at: float = float("-inf")
        self.last_reaction_at: float = float("-inf")

    def gesture(self, mood: str) -> dict | None:
        spec = self.persona.gestures.get(mood)
        if spec is None:
            log.info("no gesture configured for mood %s", mood)
            return None
        tool, arg = spec
        args = {"emotion": arg} if tool == "play_emotion" else {"direction": arg}
        log.info("gesture mood=%s -> %s %s", mood, tool, json.dumps(args))
        return call_item(tool, args)

    def _gated(self, mood: str | None, now: float) -> list[dict]:
        if mood is None or now - self.last_at < self.min_gap_s:
            return []
        item = self.gesture(mood)
        if item is None:
            return []
        self.last_at = now
        return [item]

    def for_addressed(self, text: str, now: float) -> list[dict]:
        if now - self.last_at < self.min_gap_s:
            return []
        mood: str | None = None
        if self.mood_classifier is not None:
            try:
                mood = self.mood_classifier(text)
            except Exception as e:  # a dead classify slot must not cost the gesture
                log.warning("mood classifier failed (%s); using rules", type(e).__name__)
        if mood not in self.persona.gestures:
            mood = addressed_mood(text)
        return self._gated(mood, now)

    def for_table(self, text: str, now: float) -> list[dict]:
        """A reaction to a line not spoken to Rocky, or nothing.

        Both gaps are checked before any call is made, so a rate-limited turn costs no
        round trip. Rules first; the one-token reaction call only for lines long enough
        to be narration and only when no rule matched.
        """
        if now - self.last_at < self.min_gap_s or now - self.last_reaction_at < self.reaction_gap_s:
            return []
        mood = table_mood(text)
        if (mood is None and self.reaction_classifier is not None
                and len(text.split()) >= MIN_REACTION_WORDS):
            try:
                answer = self.reaction_classifier(text)
            except Exception as e:  # never let a reaction cost anything
                log.warning("reaction classifier failed (%s); no reaction", type(e).__name__)
                answer = None
            mood = answer if answer in REACTION_MOODS else None
            log.info("reaction call answered %r", answer)
        items = self._gated(mood, now)
        if items:
            self.last_reaction_at = now
        return items

    def cue(self, mood: str, now: float) -> list[dict]:
        item = self.gesture(mood)
        if item is None:
            return []
        self.last_at = now
        return [item]
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_motion.py -v`
Expected: 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add familiar_brain/motion.py tests/test_motion.py
git commit -m "feat(motion): choreographer, table reactions with a reaction call, addressed mood rules"
```

---

### Task 4: One-token mood and reaction calls on the classify slot

**Files:**
- Modify: `familiar_brain/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_llm.py` already has a `fake` fixture yielding a `FakeLLM`. Append:

```python
def test_mood_is_one_token_on_the_classify_slot_and_validated(fake):
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    fake.chat_replies.append("Thoughtful.")
    assert c.mood("Rocky, what do you make of him?", moods=("attentive", "thoughtful")) == "thoughtful"
    body = fake.requests[-1]["body"]
    assert body["id_slot"] == 1 and body["max_tokens"] == 4 and body["reasoning_effort"] == "none"
    assert "attentive, thoughtful" in body["messages"][0]["content"]
    assert "spoken to" in body["messages"][0]["content"]
    fake.chat_replies.append("banana")
    assert c.mood("anything", moods=("attentive", "thoughtful")) is None


def test_reaction_prompt_is_selectable_and_none_is_a_valid_answer(fake):
    from familiar_brain.llm import REACTION_INSTRUCTION
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    fake.chat_replies.append("none")
    moods = ("amused", "alarmed", "sad", "scared", "none")
    assert c.mood("You walk for an hour along the road.", moods=moods, prompt=REACTION_INSTRUCTION) == "none"
    content = fake.requests[-1]["body"]["messages"][0]["content"]
    assert "not spoken to it" in content and "Most lines are none" in content
    fake.chat_replies.append("Alarmed")
    assert c.mood("The priest pulls back his hood: it is the king.", moods=moods, prompt=REACTION_INSTRUCTION) == "alarmed"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_llm.py -v`
Expected: AttributeError, no `mood`.

- [ ] **Step 3: Add to `familiar_brain/llm.py`**

Add the constants after `CLASSIFY_INSTRUCTION`:

```python
MOOD_INSTRUCTION = (
    "You choose the facial expression of a small creature that has just been spoken to at a "
    "game of Dungeons and Dragons. Read the line and answer with exactly one word from this "
    "list: {moods}. Choose attentive when nothing else clearly fits."
)

# The reaction call. The line was not spoken to Rocky; it is usually the Dungeon Master
# narrating or a player talking. The list passed in ends with "none" and the wording
# pushes toward it, for the same reason CLASSIFY_INSTRUCTION pushes toward IGNORE: a
# creature that reacts to everything reacts to nothing.
REACTION_INSTRUCTION = (
    "You choose the reaction of a small creature listening at a game of Dungeons and "
    "Dragons. The line was not spoken to it; it is usually the Dungeon Master narrating or "
    "a player talking. Answer with exactly one word from this list: {moods}.\n"
    "amused: the line is funny, or a triumph for the party.\n"
    "alarmed: a shocking twist, a sudden reveal, a betrayal.\n"
    "sad: grief, loss, a farewell, something sorrowful.\n"
    "scared: danger or a threat to the party, a monster, a trap springing.\n"
    "none: ordinary description, travel, rules talk, dice, table logistics.\n"
    "Most lines are none."
)
```

Add the method to `LLMClient`:

```python
    def mood(self, text: str, *, moods: tuple[str, ...], prompt: str = MOOD_INSTRUCTION) -> str | None:
        """One word from `moods` for the gesture that goes out with a turn.

        Same slot, budget and reasoning setting as classify: 59 ms measured live for one
        token with reasoning off (docs/versions.md). `prompt` selects the instruction;
        motion.py uses MOOD_INSTRUCTION for addressed turns and REACTION_INSTRUCTION for
        table lines. Returns None for anything outside the list so the caller falls back
        to its keyword rules.
        """
        body = {
            "messages": [
                {"role": "system", "content": prompt.format(moods=", ".join(moods))},
                {"role": "user", "content": text},
            ],
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
            "id_slot": self.classify_slot,
            "reasoning_effort": "none",
        }
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.classify_timeout_s)
        r.raise_for_status()
        word = r.json()["choices"][0]["message"]["content"].strip().lower().strip(".,!\"'")
        return word if word in moods else None
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add familiar_brain/llm.py tests/test_llm.py
git commit -m "feat(llm): one-token mood and reaction calls on the classify slot"
```

---

### Task 5: State and config for cues, tracking and modes

**Files:**
- Modify: `familiar_brain/state.py`, `familiar_brain/config.py`, `config/brain.toml`
- Test: `tests/test_state.py` (create), `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_state.py`:

```python
from familiar_brain.state import BrainState


def test_cue_is_consumed_once():
    st = BrainState()
    assert st.consume_cue() is None
    st.set_cue("asleep")
    assert st.consume_cue() == "asleep" and st.consume_cue() is None


def test_fresh_state_is_not_tracking_and_in_familiar_mode():
    st = BrainState()
    assert st.tracking is False and st.mode == "familiar" and st.mode_set_at is None
    assert st.last_glance_at == float("-inf")
```

`tests/test_config.py` has no minimal TOML constant; add this one at module top, after the imports:

```python
MINIMAL = """
[server]
host = "127.0.0.1"
port = 0
[upstream]
base_url = "http://127.0.0.1:1/v1"
generate_slot = 0
classify_slot = 1
classify_timeout_s = 0.6
[gate]
follow_up_window_s = 8
[paths]
persona = "p.yaml"
sessions_dir = "s"
[knowledge]
db_path = "k.db"
notes_dir = "n"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
"""
```

Then append:

```python
def test_robot_and_motion_sections_are_optional_with_defaults(tmp_path):
    p = tmp_path / "brain.toml"
    p.write_text(MINIMAL)
    cfg = load_config(p)
    assert cfg.robot.daemon_url == "" and cfg.robot.doa_timeout_s == 0.15
    assert cfg.motion.min_gesture_gap_s == 6.0 and cfg.motion.reaction_gap_s == 15.0
    assert cfg.motion.glance_gap_s == 12.0
    assert cfg.motion.tracking_linger_s == 4.0 and cfg.motion.mode_timeout_s == 300.0


def test_robot_and_motion_sections_load_when_present(tmp_path):
    p = tmp_path / "brain.toml"
    p.write_text(MINIMAL + '\n[robot]\ndaemon_url = "http://10.0.0.100:8000"\n[motion]\nglance_gap_s = 20\nreaction_gap_s = 30\n')
    cfg = load_config(p)
    assert cfg.robot.daemon_url == "http://10.0.0.100:8000"
    assert cfg.motion.glance_gap_s == 20.0 and cfg.motion.reaction_gap_s == 30.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_state.py tests/test_config.py -v`
Expected: AttributeError on `consume_cue`, `tracking`, `cfg.robot`.

- [ ] **Step 3: Extend `BrainState` in `familiar_brain/state.py`**

Add to `__init__` after `self.follow_ups = 0`:

```python
        # A gesture that must go out with the next response regardless of rate limits:
        # the mute and unmute cues. Set by handlers, consumed by the responses route.
        self._pending_cue: str | None = None
        # What the brain last told the robot about head tracking. Toggles go out only
        # on a change, so the app's MovementManager is never told the same thing twice.
        self.tracking = False
        self.last_glance_at: float = float("-inf")
        # Reply mode, see modes.py. "familiar" is the two-sentence table default.
        self.mode = "familiar"
        self.mode_set_at: float | None = None
```

Add the methods:

```python
    def set_cue(self, mood: str) -> None:
        with self._lock:
            self._pending_cue = mood

    def consume_cue(self) -> str | None:
        with self._lock:
            cue, self._pending_cue = self._pending_cue, None
            return cue
```

- [ ] **Step 4: Add config sections in `familiar_brain/config.py`**

Add two dataclasses before `Config`:

```python
@dataclass(frozen=True)
class Robot:
    # The robot daemon, read-only: the brain GETs /api/state/doa to aim glances and
    # never writes to it (the conversation app owns the head). Empty disables DoA.
    daemon_url: str = ""
    doa_timeout_s: float = 0.15


@dataclass(frozen=True)
class Motion:
    # No two gestures closer than this, whatever their source.
    min_gesture_gap_s: float = 6.0
    # No two reactions to table lines closer than this; also skips the reaction call.
    reaction_gap_s: float = 15.0
    glance_gap_s: float = 12.0
    # How long after the follow-up window closes head tracking stays on.
    tracking_linger_s: float = 4.0
    # Thinking mode ends after this long without an addressed turn.
    mode_timeout_s: float = 300.0
```

Add `robot: Robot = Robot()` and `motion: Motion = Motion()` as the last two fields of `Config`, and in `load_config` add:

```python
        robot=Robot(**raw.get("robot", {})),
        motion=Motion(**raw.get("motion", {})),
```

- [ ] **Step 5: Add the sections to `config/brain.toml`**

```toml
[robot]
daemon_url = "http://10.0.0.100:8000"
doa_timeout_s = 0.15

[motion]
min_gesture_gap_s = 6
reaction_gap_s = 15
glance_gap_s = 12
tracking_linger_s = 4
mode_timeout_s = 300
```

- [ ] **Step 6: Run the suite**

Run: `uv run pytest -q`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add familiar_brain/state.py familiar_brain/config.py config/brain.toml tests/test_state.py tests/test_config.py
git commit -m "feat(state): pending cue, tracking and mode state; [robot] and [motion] config"
```

---

### Task 6: Persona fixed blocks for sight and thinking mode

**Files:**
- Modify: `familiar_brain/persona.py`
- Test: `tests/test_persona.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_persona.py`:

```python
def test_sight_block_is_fixed_and_forbids_identifying_people(tmp_path: Path):
    p = tmp_path / "persona.yaml"; p.write_text(GOOD)
    prompt = render_system_prompt(load_persona(p))
    assert "camera tool" in prompt and "Never name or identify a person" in prompt


def test_thinking_mode_changes_only_the_length_block(tmp_path: Path):
    p = tmp_path / "persona.yaml"; p.write_text(GOOD)
    per = load_persona(p)
    default, thinking = render_system_prompt(per), render_system_prompt(per, mode="thinking")
    assert default.endswith("two spoken sentences unless explicitly asked for more.")
    assert "up to six sentences" in thinking and "two spoken sentences" not in thinking
    assert default.rsplit("\n\n", 1)[0] == thinking.rsplit("\n\n", 1)[0]
    assert render_system_prompt(per, mode="nonsense") == default
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_persona.py -v`
Expected: the sight test fails on the missing text; the mode test fails with TypeError on `mode`.

- [ ] **Step 3: Modify `render_system_prompt` in `familiar_brain/persona.py`**

Add before the function:

```python
LENGTH_BLOCKS = {
    "familiar": "Keep every reply to at most two spoken sentences unless explicitly asked for more.",
    "thinking": (
        "The speaker has asked you to think alongside them. Reason out loud in plain spoken "
        "language, up to six sentences: name the options you see, say which you favour and why, "
        "and finish with one question back or one concrete suggestion. Stay in character; you "
        "are still a creature, not a consultant."
    ),
}
```

Change the signature to `def render_system_prompt(p: Persona, context: str = "", mode: str = "familiar") -> str:` and extend the docstring with: `mode` selects the length block; an unknown mode falls back to "familiar".

After `tool_block` add:

```python
    sight_block = (
        "You have a camera. When asked what you see, who is at the table, how many people there "
        "are, what someone is holding, or to read dice or a card, call the camera tool with a "
        "short question, then describe only what the picture shows. Never name or identify a "
        "person; say 'the one on your left' or 'the person holding the cards'. If the picture is "
        "unclear, say so rather than guess.\n"
    )
```

Replace the last three lines with:

```python
    length_block = LENGTH_BLOCKS.get(mode, LENGTH_BLOCKS["familiar"])
    prompt = "\n\n".join([persona_block, off_limits_block + dm_block + tool_block + sight_block, length_block])
    return f"{context}\n\n{prompt}" if context else prompt
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_persona.py -v`
Expected: all PASS, including `test_persona_cannot_override_fixed_blocks`.

- [ ] **Step 5: Commit**

```bash
git add familiar_brain/persona.py tests/test_persona.py
git commit -m "feat(persona): fixed sight block and a thinking-mode length block"
```

---

## Part C: Attention

### Task 7: robot.py, the read-only daemon client

**Files:**
- Create: `familiar_brain/robot.py`
- Test: `tests/test_robot.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_robot.py`:

```python
import math
from familiar_brain.robot import RobotDaemon, direction_from_angle


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status
    def raise_for_status(self):
        if self.status != 200:
            raise RuntimeError(f"http {self.status}")
    def json(self):
        return self.payload


def test_direction_from_angle_uses_the_daemon_convention():
    """0 = left, pi/2 = front, pi = right (reachy_mini io/protocol.py DoaSnapshot)."""
    assert direction_from_angle(0.2) == "left"
    assert direction_from_angle(math.pi / 2) == "front"
    assert direction_from_angle(math.pi - 0.2) == "right"
    assert direction_from_angle(math.pi / 2 + 0.3) == "front"     # inside the dead zone


def test_doa_reads_angle_and_speech_flag():
    seen = []
    def get(url, timeout):
        seen.append((url, timeout)); return _Resp({"angle": 0.3, "speech_detected": True})
    d = RobotDaemon("http://robot:8000/", timeout_s=0.15, get=get)
    assert d.doa() == (0.3, True)
    assert seen == [("http://robot:8000/api/state/doa", 0.15)]


def test_doa_degrades_to_none_on_null_error_or_timeout():
    assert RobotDaemon("http://r", get=lambda u, timeout: _Resp(None)).doa() is None
    assert RobotDaemon("http://r", get=lambda u, timeout: _Resp({}, status=503)).doa() is None
    def boom(u, timeout): raise TimeoutError("slow")
    assert RobotDaemon("http://r", get=boom).doa() is None
    assert RobotDaemon("", get=boom).doa() is None       # unconfigured: never even calls
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_robot.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Create `familiar_brain/robot.py`**

```python
"""Read-only client for the Reachy Mini daemon on the robot.

The conversation app owns the head, the antennas and the camera. The brain never
commands the daemon; it only reads the microphone array's direction of arrival so a
glance can be aimed at whoever just spoke. Every failure degrades to None inside the
gate's latency budget, because a glance is never worth a late reply.
"""
from __future__ import annotations

import logging
import math
from typing import Callable

import requests

log = logging.getLogger("familiar.robot")

# Below pi/2 minus this is left, above pi/2 plus this is right; the middle is front.
DEAD_ZONE_RAD = 0.35


def direction_from_angle(angle: float, dead_zone: float = DEAD_ZONE_RAD) -> str:
    """Map a DoA angle (0 left, pi/2 front, pi right) to a move_head direction."""
    if angle < math.pi / 2 - dead_zone:
        return "left"
    if angle > math.pi / 2 + dead_zone:
        return "right"
    return "front"


class RobotDaemon:
    def __init__(self, base_url: str, *, timeout_s: float = 0.15,
                 get: Callable[..., requests.Response] = requests.get):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.get = get

    def doa(self) -> tuple[float, bool] | None:
        """(angle_rad, speech_detected) from GET /api/state/doa, or None."""
        if not self.base_url:
            return None
        try:
            r = self.get(f"{self.base_url}/api/state/doa", timeout=self.timeout_s)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            log.debug("doa unavailable: %s", type(e).__name__)
            return None
        if not isinstance(payload, dict) or "angle" not in payload:
            return None
        return float(payload["angle"]), bool(payload.get("speech_detected", False))
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_robot.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add familiar_brain/robot.py tests/test_robot.py
git commit -m "feat(robot): read-only daemon client for sound direction"
```

---

### Task 8: attention.py, tracking toggles and glances per turn

**Files:**
- Create: `familiar_brain/attention.py`
- Test: `tests/test_attention.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_attention.py`:

```python
import json
import math

from familiar_brain.attention import Attention
from familiar_brain.gate import Decision
from familiar_brain.state import BrainState


def make(doa=lambda: None, window=8.0, linger=4.0, glance_gap=12.0):
    st = BrainState()
    return st, Attention(state=st, doa=doa, follow_up_window_s=window, linger_s=linger, glance_gap_s=glance_gap)


def _calls(items):
    return [(i["name"], json.loads(i["arguments"])) for i in items]


def test_tracking_turns_on_when_addressed_and_off_after_the_window_plus_linger():
    st, att = make()
    st.mark_named(100.0)
    assert _calls(att.items(Decision.ADDRESSED, 100.0)) == [("head_tracking", {"enabled": True})]
    assert att.items(Decision.ADDRESSED, 103.0) == []                      # no change, no repeat
    assert att.items(Decision.IGNORE, 110.0) == []                         # 8 + 4 = 12 s of linger
    assert _calls(att.items(Decision.IGNORE, 112.5)) == [("head_tracking", {"enabled": False})]
    assert st.tracking is False


def test_glances_only_while_tracking_is_off_and_rate_limited():
    st, att = make(doa=lambda: (0.1, True))
    assert _calls(att.items(Decision.IGNORE, 0.0)) == [("move_head", {"direction": "left"})]
    assert att.items(Decision.AMBIENT, 5.0) == []                          # inside glance gap
    assert _calls(att.items(Decision.AMBIENT, 13.0)) == [("move_head", {"direction": "left"})]
    st.mark_named(20.0)
    assert _calls(att.items(Decision.ADDRESSED, 20.0)) == [("head_tracking", {"enabled": True})]
    assert att.items(Decision.IGNORE, 21.0) == []                          # tracking owns the head


def test_no_doa_means_no_glance():
    _, att = make(doa=lambda: None)
    assert att.items(Decision.IGNORE, 0.0) == []


def test_muted_never_tracks_and_never_glances():
    st, att = make(doa=lambda: (math.pi, True))
    st.mark_named(0.0); att.items(Decision.ADDRESSED, 0.0)
    st.muted = True
    assert _calls(att.items(Decision.IGNORE, 1.0)) == [("head_tracking", {"enabled": False})]
    assert att.items(Decision.IGNORE, 30.0) == []
    assert att.items(Decision.IGNORE, 60.0) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_attention.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Create `familiar_brain/attention.py`**

```python
"""Where Rocky looks, decided once per turn (design 4.7).

Head tracking is the conversation app's daemon-side face tracker. It is on while
Rocky is being spoken to and for a short linger after the follow-up window closes,
and off otherwise, because move_head is invisible while the tracker owns the head at
weight 1. While it is off, an IGNORE or AMBIENT line earns a glance toward the sound
direction the daemon last measured, rate limited so Rocky watches the table rather
than twitching at every line.

Toggles go out only on a change: the app's MovementManager keeps its own flag and a
repeated head_tracking(true) would reset its speaking handoff.
"""
from __future__ import annotations

import logging
from typing import Callable

from familiar_brain.gate import Decision
from familiar_brain.motion import glance_item, tracking_item
from familiar_brain.robot import direction_from_angle
from familiar_brain.state import BrainState

log = logging.getLogger("familiar.attention")

DoaReader = Callable[[], tuple[float, bool] | None]


class Attention:
    def __init__(self, *, state: BrainState, doa: DoaReader, follow_up_window_s: float,
                 linger_s: float, glance_gap_s: float):
        self.state = state
        self.doa = doa
        self.window = follow_up_window_s
        self.linger = linger_s
        self.glance_gap = glance_gap_s

    def wants_tracking(self, decision: Decision, now: float) -> bool:
        if self.state.muted:
            return False
        if decision is Decision.ADDRESSED:
            return True
        named = self.state.last_named_at
        return named is not None and now < named + self.window + self.linger

    def items(self, decision: Decision, now: float) -> list[dict]:
        out: list[dict] = []
        want = self.wants_tracking(decision, now)
        if want != self.state.tracking:
            self.state.tracking = want
            log.info("head tracking %s", "on" if want else "off")
            out.append(tracking_item(want))
        if want or self.state.muted:
            return out
        if decision not in (Decision.IGNORE, Decision.AMBIENT):
            return out
        if now - self.state.last_glance_at < self.glance_gap:
            return out
        reading = self.doa()
        if reading is None:
            return out
        angle, _speech = reading
        direction = direction_from_angle(angle)
        self.state.last_glance_at = now
        log.info("glance %s (doa %.2f rad)", direction, angle)
        out.append(glance_item(direction))
        return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_attention.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add familiar_brain/attention.py tests/test_attention.py
git commit -m "feat(attention): tracking toggles while addressed, DoA glances while not"
```

---

## Part D: Modes and sight plumbing

### Task 9: modes.py

**Files:**
- Create: `familiar_brain/modes.py`
- Test: `tests/test_modes.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_modes.py`:

```python
from familiar_brain.modes import FAMILIAR, THINKING, update_mode
from familiar_brain.state import BrainState


def test_enter_and_exit_phrases_on_addressed_turns_only():
    st = BrainState()
    assert update_mode(st, "Rocky, think with me about the cult", addressed=True, now=0.0, timeout_s=300) == THINKING
    assert update_mode(st, "let's think about lunch", addressed=False, now=1.0, timeout_s=300) == THINKING
    assert update_mode(st, "Rocky, what about the docks?", addressed=True, now=2.0, timeout_s=300) == THINKING
    assert update_mode(st, "Rocky, thanks, that's all", addressed=True, now=3.0, timeout_s=300) == FAMILIAR


def test_thinking_times_out_without_addressed_turns():
    st = BrainState()
    update_mode(st, "Rocky, help me think this through", addressed=True, now=0.0, timeout_s=300)
    update_mode(st, "Rocky, and?", addressed=True, now=100.0, timeout_s=300)   # keeps it alive
    assert update_mode(st, "table talk", addressed=False, now=399.0, timeout_s=300) == THINKING
    assert update_mode(st, "table talk", addressed=False, now=401.0, timeout_s=300) == FAMILIAR


def test_exit_phrases_win_over_enter_phrases_in_one_line():
    st = BrainState()
    assert update_mode(st, "Rocky, quick answer, no need to think it through", addressed=True, now=0.0, timeout_s=300) == FAMILIAR
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_modes.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Create `familiar_brain/modes.py`**

```python
"""Reply modes: the two-sentence familiar, or a thinking partner when asked.

The mode changes one fixed block of the system prompt (persona.LENGTH_BLOCKS) and
nothing else. It is opened and closed by phrases on ADDRESSED turns, by the companion
UI through POST /api/mode, and it expires after mode_timeout_s without an addressed
turn so a table never inherits a chatty familiar from an hour ago.
"""
from __future__ import annotations

import logging
import re

from familiar_brain.state import BrainState

log = logging.getLogger("familiar.modes")

FAMILIAR = "familiar"
THINKING = "thinking"
MODES = (FAMILIAR, THINKING)

ENTER = re.compile(
    r"\b(think (it |this |that )?(through )?with me|let'?s think|help me think|thought partner"
    r"|talk (it|this|that) through|what are (my|our) options)\b", re.I)
EXIT = re.compile(
    r"\b(that'?s (all|enough|it)|thank you|thanks|quick answer|short answer|back to normal"
    r"|stop thinking)\b", re.I)


def update_mode(state: BrainState, text: str, *, addressed: bool, now: float, timeout_s: float) -> str:
    if (state.mode == THINKING and state.mode_set_at is not None
            and now - state.mode_set_at > timeout_s):
        log.info("thinking mode timed out")
        state.mode = FAMILIAR
    if not addressed:
        return state.mode
    if EXIT.search(text):
        if state.mode != FAMILIAR:
            log.info("thinking mode closed by phrase")
        state.mode = FAMILIAR
    elif ENTER.search(text):
        if state.mode != THINKING:
            log.info("thinking mode opened by phrase")
        state.mode = THINKING
    if state.mode == THINKING:
        state.mode_set_at = now
    return state.mode
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_modes.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add familiar_brain/modes.py tests/test_modes.py
git commit -m "feat(modes): thinking mode opened and closed by phrase, with timeout"
```

---

### Task 10: Continuation detection for camera frames, echoes and the greeting

**Files:**
- Modify: `familiar_brain/responses_api.py`
- Test: `tests/test_responses_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_responses_api.py`:

```python
from familiar_brain.responses_api import continuation_kind, GREETING_MARKER

IMG = {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/AAA="}


def test_camera_frame_as_last_item_is_a_sight_continuation():
    req = {"input": [
        {"role": "user", "content": [{"type": "input_text", "text": "Rocky, what do you see?"}]},
        {"type": "function_call", "call_id": "call_1", "name": "camera", "arguments": "{\"question\":\"scene\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "{\"image_attached\": true}"},
        {"type": "message", "role": "user", "content": [IMG]},
    ]}
    assert continuation_kind(req) == "sight"


def test_camera_result_without_a_frame_is_still_sight_so_the_model_can_say_it_failed():
    req = {"input": [
        {"role": "user", "content": "Rocky, look"},
        {"type": "function_call", "call_id": "call_2", "name": "camera", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_2", "output": "{\"error\": \"Camera is disabled\"}"},
    ]}
    assert continuation_kind(req) == "sight"


def test_other_tool_results_are_echoes():
    req = {"input": [
        {"role": "user", "content": "the door creaks"},
        {"type": "function_call", "call_id": "call_3", "name": "idle_do_nothing", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_3", "output": "{\"status\": \"idle\"}"},
    ]}
    assert continuation_kind(req) == "echo"


def test_ordinary_utterances_and_the_greeting_prompt():
    assert continuation_kind({"input": [{"role": "user", "content": "Rocky, hello"}]}) is None
    assert continuation_kind({"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}, IMG]}]}) is None
    assert continuation_kind({"input": []}) is None
    greeting = {"input": [{"role": "user", "content": [{"type": "input_text", "text": GREETING_MARKER + " with a brief greeting"}]}]}
    assert continuation_kind(greeting) == "greeting"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_responses_api.py -v`
Expected: ImportError on `continuation_kind`.

- [ ] **Step 3: Add to `familiar_brain/responses_api.py`**

```python
CAMERA_TOOL = "camera"
# The conversation app opens a session by asking the model for a greeting with this
# exact prompt (prompts.py DEFAULT_GREETING_PROMPT). It is not a table utterance.
GREETING_MARKER = "Start the conversation now"


def continuation_kind(req: dict) -> str | None:
    """Classify a request that is not a fresh utterance from the table.

    None:       an ordinary utterance; run the gate.
    "sight":    the client just attached a camera frame, or returned the camera tool's
                result; the next generation must see it, so the gate is bypassed.
    "echo":     the client returned the output of another robot-side tool (the app
                requests a response after idle_do_nothing and camera, which keep
                needs_response=True). Answer with nothing; it cannot loop.
    "greeting": the app's session-open greeting prompt.
    Shapes from huggingface_realtime.py (app 1.0.1) and docs/protocol-notes.md 4 and 6.
    """
    items = req.get("input") or []
    last = items[-1] if items else None
    if not isinstance(last, dict):
        return None
    if last.get("role") == "user":
        content = last.get("content")
        if isinstance(content, list):
            kinds = {p.get("type") for p in content if isinstance(p, dict)}
            if "input_image" in kinds and not kinds & {"input_text", "text"}:
                return "sight"
        return "greeting" if GREETING_MARKER in _text_of(content) else None
    if last.get("type") == "function_call_output":
        names = {i.get("call_id"): i.get("name") for i in items
                 if isinstance(i, dict) and i.get("type") == "function_call"}
        return "sight" if names.get(last.get("call_id")) == CAMERA_TOOL else "echo"
    return None
```

`_text_of` is the module's existing helper that flattens a content string or part list to text (read `latest_user_text` to confirm its name before using it; if the helper is inlined there, extract it).

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_responses_api.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add familiar_brain/responses_api.py tests/test_responses_api.py
git commit -m "feat(responses): classify camera, echo and greeting continuations"
```

---

## Part E: Wiring

### Task 11: app.py, prefaces, reactions, continuations, modes

**Files:**
- Modify: `familiar_brain/app.py`, `tests/test_app_integration.py`
- Test: `tests/test_app_expression.py`, `tests/test_app_sight.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_app_expression.py`:

```python
import json
from pathlib import Path

import pytest

from familiar_brain import indexing
from familiar_brain.app import create_app
from familiar_brain.retrieval import Store
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm_server import FakeLLM

PERSONA = """name: Rocky
aliases: [rocky]
tts_speaker: Aiden
manner: dry
gestures:
  attentive: play_emotion:attentive
  amused: play_emotion:happy
  alarmed: play_emotion:surprised
  thoughtful: play_emotion:thinking
  sad: play_emotion:sad
  scared: play_emotion:scared
  asleep: play_emotion:sleepy
  awake: play_emotion:welcoming
"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(indexing, "open_store", lambda cfg: Store(tmp_path / "k.db", embedder=FakeEmbedder()))
    persona = tmp_path / "persona.yaml"; persona.write_text(PERSONA)
    cfg = tmp_path / "brain.toml"
    cfg.write_text(f"""
[server]
host = "127.0.0.1"
port = 0
[upstream]
base_url = "{fake.base_url}"
generate_slot = 0
classify_slot = 1
classify_timeout_s = 1.0
[gate]
follow_up_window_s = 8
[paths]
persona = "{persona.as_posix()}"
sessions_dir = "{(tmp_path / 'sessions').as_posix()}"
[knowledge]
db_path = "{(tmp_path / 'k.db').as_posix()}"
notes_dir = "{(tmp_path / 'notes').as_posix()}"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
[motion]
min_gesture_gap_s = 0
reaction_gap_s = 0
glance_gap_s = 0
""")
    app = create_app(cfg)
    app.config["TESTING"] = True
    yield app.test_client(), fake, tmp_path, app
    fake.close()


def _calls(resp) -> list[tuple[str, dict]]:
    out = []
    for l in resp.get_data(as_text=True).splitlines():
        if l.startswith("data: "):
            ev = json.loads(l[6:]); item = ev.get("item") or {}
            if ev["type"] == "response.output_item.done" and item.get("type") == "function_call":
                out.append((item["name"], json.loads(item["arguments"])))
    return out


def _types(resp) -> list[str]:
    return [json.loads(l[6:])["type"] for l in resp.get_data(as_text=True).splitlines() if l.startswith("data: ")]


def _say(client, text):
    return client.post("/v1/responses", json={"model": "familiar", "stream": True, "tools": [],
                                              "input": [{"role": "user", "content": text}]})


def test_addressed_turn_is_prefaced_by_tracking_then_a_gesture_then_speech(env):
    client, fake, _, _ = env
    fake.chat_replies.append("thoughtful")      # the mood call; classify is skipped on a named line
    resp = _say(client, "Rocky, what do you make of him?")
    calls = _calls(resp)
    assert calls == [("head_tracking", {"enabled": True}), ("play_emotion", {"emotion": "thinking"})]
    types = _types(resp)
    assert types.index("response.output_item.done") < types.index("response.output_text.delta")


def test_ambient_outburst_gets_a_rule_gesture_and_no_speech_or_idle(env):
    client, fake, _, _ = env
    fake.chat_replies.append("AMBIENT")         # the gate; "Natural twenty!" hits a rule, so no reaction call
    resp = _say(client, "Natural twenty!")
    assert _calls(resp) == [("play_emotion", {"emotion": "happy"})]
    assert "response.output_text.delta" not in _types(resp)


def test_narration_twist_gets_a_surprised_reaction_from_the_reaction_call(env):
    client, fake, _, _ = env
    fake.chat_replies.append("IGNORE")          # the gate: narration is IGNORE (deviation E8)
    fake.chat_replies.append("alarmed")         # the reaction call
    resp = _say(client, "The priest pulls back his hood. It is the king, and he has been dead a year.")
    assert _calls(resp) == [("play_emotion", {"emotion": "surprised"})]
    assert "response.output_text.delta" not in _types(resp)
    assert "not spoken to it" in fake.requests[-1]["body"]["messages"][0]["content"]


def test_sad_narration_gets_a_sad_reaction(env):
    client, fake, _, _ = env
    fake.chat_replies.append("IGNORE")
    fake.chat_replies.append("sad")
    resp = _say(client, "She folds the letter and does not read the rest aloud.")
    assert _calls(resp) == [("play_emotion", {"emotion": "sad"})]


def test_plain_narration_answered_none_still_idles(env):
    client, fake, _, _ = env
    fake.chat_replies.append("IGNORE")
    fake.chat_replies.append("none")
    assert _calls(_say(client, "You walk for an hour along the river road.")) == [("idle_do_nothing", {})]


def test_short_ignored_line_makes_no_reaction_call(env):
    client, fake, _, _ = env
    fake.chat_replies.append("IGNORE")
    before = len(fake.requests)
    assert _calls(_say(client, "the door creaks")) == [("idle_do_nothing", {})]
    assert len(fake.requests) - before == 1     # the gate only


def test_tracking_is_switched_off_once_the_window_and_linger_have_passed(env):
    client, fake, _, app = env
    _say(client, "Rocky, hello")
    bs = app.extensions["brain_state"]
    bs.last_named_at -= 60; bs.last_spoke_at = (bs.last_spoke_at or 0) - 60
    fake.chat_replies.append("IGNORE")
    assert ("head_tracking", {"enabled": False}) in _calls(_say(client, "moving on"))


def test_mute_and_unmute_cue_sleep_and_wake_gestures_and_muted_never_reacts(env):
    client, fake, _, _ = env
    client.post("/api/mute")
    assert _calls(_say(client, "Rocky, hello")) == [("play_emotion", {"emotion": "sleepy"})]
    before = len(fake.requests)
    assert _calls(_say(client, "The priest pulls back his hood and it is the king.")) == [("idle_do_nothing", {})]
    assert len(fake.requests) == before         # muted: no gate call, no reaction call
    client.post("/api/unmute")
    fake.chat_replies.append("IGNORE")
    assert _calls(_say(client, "table talk")) == [("play_emotion", {"emotion": "welcoming"})]


def test_mode_api_and_phrase_change_the_length_block_sent_upstream(env):
    client, fake, _, _ = env
    _say(client, "Rocky, think with me about the cult")
    assert "up to six sentences" in fake.requests[-1]["body"]["instructions"]
    assert client.get("/api/state").get_json()["mode"] == "thinking"
    assert client.post("/api/mode", json={"mode": "familiar"}).status_code == 200
    _say(client, "Rocky, and the docks?")
    assert "two spoken sentences" in fake.requests[-1]["body"]["instructions"]
    assert client.post("/api/mode", json={"mode": "loud"}).status_code == 400
```

`tests/test_app_sight.py`:

```python
import json
from pathlib import Path

import pytest

from tests.test_app_expression import env, _calls, _types, _say  # noqa: F401  (fixture reuse)

IMG = {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/AAA="}


def _camera_round_trip(text, output):
    return {"model": "familiar", "stream": True, "tools": [], "input": [
        {"role": "user", "content": [{"type": "input_text", "text": text}]},
        {"type": "function_call", "call_id": "call_c1", "name": "camera", "arguments": "{\"question\":\"scene\"}"},
        {"type": "function_call_output", "call_id": "call_c1", "output": output},
        {"type": "message", "role": "user", "content": [IMG]},
    ]}


def test_camera_frame_reaches_upstream_and_is_never_logged(env):
    client, fake, tmp, _ = env
    client.post("/api/session/start")
    _say(client, "Rocky, what do you see?")
    fake.responses_events.append([
        {"type": "response.output_text.delta", "delta": "Four mortals and a great many dice."},
        {"type": "response.completed", "response": {"id": "r", "output": []}},
    ])
    resp = client.post("/v1/responses", json=_camera_round_trip("Rocky, what do you see?", "{\"image_attached\": true}"))
    assert "response.output_text.delta" in _types(resp)
    sent = fake.requests[-1]["body"]["input"]
    assert sent[-1]["content"][0]["type"] == "input_image" and sent[-1]["content"][0]["image_url"].startswith("data:image/jpeg")
    log = next((tmp / "sessions").glob("*.jsonl")).read_text()
    assert "data:image" not in log and log.count("\n") == 1       # only the spoken question was logged


def test_camera_continuation_is_silent_while_muted(env):
    client, fake, _, _ = env
    _say(client, "Rocky, what do you see?")
    client.post("/api/mute")
    resp = client.post("/v1/responses", json=_camera_round_trip("Rocky, what do you see?", "{\"image_attached\": true}"))
    assert "response.output_text.delta" not in _types(resp)
    assert ("play_emotion", {"emotion": "sleepy"}) in _calls(resp)


def test_tool_result_echo_gets_an_empty_response_and_no_log_line(env):
    client, fake, tmp, _ = env
    client.post("/api/session/start")
    fake.chat_replies.append("IGNORE")
    _say(client, "the door creaks")
    resp = client.post("/v1/responses", json={"model": "familiar", "stream": True, "input": [
        {"role": "user", "content": "the door creaks"},
        {"type": "function_call", "call_id": "call_i1", "name": "idle_do_nothing", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_i1", "output": "{\"status\": \"idle\"}"},
    ]})
    assert _calls(resp) == [] and _types(resp) == ["response.created", "response.completed"]
    assert next((tmp / "sessions").glob("*.jsonl")).read_text().count("\n") == 1


def test_greeting_prompt_is_answered_with_a_wake_gesture_and_no_open_window(env):
    client, fake, _, app = env
    fake.responses_events.append([
        {"type": "response.output_text.delta", "delta": "Evening."},
        {"type": "response.completed", "response": {"id": "r", "output": []}},
    ])
    resp = client.post("/v1/responses", json={"model": "familiar", "stream": True, "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text",
         "text": "Start the conversation now with a brief, spontaneous greeting in character."}]}]})
    assert ("play_emotion", {"emotion": "welcoming"}) in _calls(resp) and "response.output_text.delta" in _types(resp)
    assert app.extensions["brain_state"].last_named_at is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_app_expression.py tests/test_app_sight.py -v`
Expected: failures on missing `/api/mode`, missing preface items, `mode` absent from state.

- [ ] **Step 3: Rewrite the request path in `familiar_brain/app.py`**

Add imports:

```python
import time

from familiar_brain import modes
from familiar_brain.attention import Attention
from familiar_brain.llm import REACTION_INSTRUCTION
from familiar_brain.motion import REACTION_MOODS, Choreographer
from familiar_brain.robot import RobotDaemon
```

Extend the `responses_api` import with `continuation_kind, empty_response_events, idle_items, item_events, tool_call_events`.

In `create_app`, after `upstream = Upstream(...)`:

```python
    robot = RobotDaemon(cfg.robot.daemon_url, timeout_s=cfg.robot.doa_timeout_s)
    choreo = Choreographer(
        persona, min_gap_s=cfg.motion.min_gesture_gap_s, reaction_gap_s=cfg.motion.reaction_gap_s,
        clock=time.time,
        mood_classifier=lambda t: llm.mood(t, moods=tuple(persona.gestures)),
        reaction_classifier=lambda t: llm.mood(t, moods=REACTION_MOODS + ("none",), prompt=REACTION_INSTRUCTION),
    )
    attention = Attention(state=state, doa=robot.doa, follow_up_window_s=cfg.gate.follow_up_window_s,
                          linger_s=cfg.motion.tracking_linger_s, glance_gap_s=cfg.motion.glance_gap_s)
```

Change `_instructions` to skip retrieval on an empty query and pass the mode:

```python
    def _instructions(text: str, timer: TurnTimer) -> str:
        context = ""
        if text.strip():
            try:
                context = build_context(store, text, chunks_per_corpus=cfg.knowledge.chunks_per_corpus,
                                        max_chars=cfg.knowledge.max_context_chars)
            except Exception as e:  # a broken store must not cost the table a reply
                app.logger.warning("retrieval failed: %s", e)
        timer.mark("retrieval")
        return render_system_prompt(persona, context, mode=state.mode)
```

Add one helper after `_instructions`:

```python
    def _preface(text: str, decision: Decision, now: float) -> list[dict]:
        """Robot-side tool calls that go out ahead of (or instead of) speech this turn.

        Order: a pending cue (mute, unmute), then attention (tracking toggle or glance),
        then the mood of the reply on an addressed turn or a reaction to the table on
        any other. A muted brain gestures only its cue: no tracking, no reaction.
        """
        items: list[dict] = []
        cue = state.consume_cue()
        if cue:
            items += choreo.cue(cue, now)
        items += attention.items(decision, now)
        if state.muted or not text:
            return items
        if decision is Decision.ADDRESSED:
            items += choreo.for_addressed(text, now)
        else:
            items += choreo.for_table(text, now)
        return items
```

Replace the `responses` route with:

```python
    @app.post("/v1/responses")
    def responses():
        req = request.get_json(force=True) or {}
        timer = TurnTimer()
        model = str(req.get("model") or "familiar")
        now = time.time()
        kind = continuation_kind(req)
        if kind == "echo":
            timer.finish(decision="ECHO", spoke=False)
            body = "".join(sse(e) for e in empty_response_events())
            return Response(body, content_type="text/event-stream")
        if kind == "sight":
            # The frame the robot just took. Not an utterance: no lexicon, no log, no
            # follow-up accounting. Muted means eyes closed.
            text = ""
            decision = Decision.IGNORE if state.muted else Decision.ADDRESSED
            timer.mark("gate")
        elif kind == "greeting":
            # The app's session-open prompt. Answer it, but do not open the follow-up
            # window: nobody at the table has said anything yet (compare deviation E7).
            text = ""
            decision = Decision.IGNORE if state.muted else Decision.ADDRESSED
            if decision is Decision.ADDRESSED:
                state.set_cue("awake")
            timer.mark("gate")
        else:
            text = _correct_user_text(req, lexicon)
            state.remember(text)
            decision = gate.decide(state.recent_segments)
            timer.mark("gate")
            modes.update_mode(state, text, addressed=decision is Decision.ADDRESSED,
                              now=now, timeout_s=cfg.motion.mode_timeout_s)
        if not wants_stream(req):
            return _without_streaming(req, text, decision, timer, model)
        preface = _preface(text, decision, now)
        timer.mark("preface")
        if decision is not Decision.ADDRESSED:
            if kind is None:
                _log(text, decision, spoke=False)
            timer.finish(decision=decision.value, spoke=False)
            body = "".join(sse(e) for e in tool_call_events(preface or idle_items()))
            return Response(body, content_type="text/event-stream")
        outgoing = with_instructions_and_tools(req, instructions=_instructions(text, timer),
                                              extra_tools=tools.specs())

        def gen():
            spoke = False
            try:
                for ev in item_events(preface):
                    yield sse(ev)
                for frame in upstream.stream(outgoing):
                    if "response.output_text.delta" in frame:
                        spoke = True
                    yield frame
            except Exception as e:  # upstream down: degrade to silence
                app.logger.error("upstream failed: %s", e)
                for ev in idle_response_events():
                    yield sse(ev)
            finally:
                if spoke and kind is None:
                    state.mark_spoke()
                if kind is None:
                    _log(text, decision, spoke=spoke)
                timer.mark("generation")
                timer.finish(decision=decision.value if kind is None else kind.upper(), spoke=spoke)

        return Response(stream_with_context(gen()), content_type="text/event-stream")
```

The `timer.mark("preface")` line records how long the mood or reaction call took on non-addressed turns, which is what D2 and D9 say to watch at the live check. If `TurnTimer.mark` only accepts a fixed set of stage names, add `preface` to it.

Update the mute routes and state:

```python
    @app.get("/api/state")
    def api_state():
        return jsonify(muted=state.muted, mode=state.mode, tracking=state.tracking,
                       logging=state.logging_enabled, session=log.active_id,
                       recent=state.recent_segments, last_spoke_at=state.last_spoke_at)

    @app.post("/api/mute")
    def api_mute():
        state.muted = True; state.set_cue("asleep"); return jsonify(ok=True)

    @app.post("/api/unmute")
    def api_unmute():
        state.muted = False; state.logging_enabled = True; state.set_cue("awake"); return jsonify(ok=True)

    @app.post("/api/mute_and_stop_logging")
    def api_mute_stop():
        state.muted = True; state.logging_enabled = False; state.set_cue("asleep"); return jsonify(ok=True)

    @app.post("/api/mode")
    def api_mode():
        mode = str((request.get_json(force=True) or {}).get("mode", ""))
        if mode not in modes.MODES:
            return jsonify(error=f"mode must be one of {', '.join(modes.MODES)}"), 400
        state.mode = mode
        state.mode_set_at = time.time() if mode == modes.THINKING else None
        return jsonify(mode=mode)
```

- [ ] **Step 4: Run the new tests, then the whole suite**

Run: `uv run pytest tests/test_app_expression.py tests/test_app_sight.py -v && uv run pytest -q`
Expected: all new tests PASS. Two existing expectations in `tests/test_app_integration.py` change for a correct reason and must be updated, not weakened:

- `test_mock_session_speaks_only_when_addressed` asserts `names == ["idle_do_nothing"]` for every non-addressed row. Give the fixture persona (written inline in its `env` fixture) the `gestures:` block from `tests/test_app_expression.py` so gestures can fire. Then the AMBIENT row ("Natural twenty!") yields `["play_emotion"]`, the first IGNORE row after the window closes also carries `head_tracking(false)`, and the two narration rows of five words or more make a reaction call that pops the fake's queue. The fixture rows queue exactly one gate reply each, and the fake answers "IGNORE" to an unqueued reaction call, which is not a reaction mood, so the narration rows still idle. Change the non-addressed branch to:

```python
        else:
            assert "response.output_text.delta" not in _types(resp)
            expected_motion = {"AMBIENT": {"play_emotion"},
                               "IGNORE": {"idle_do_nothing", "head_tracking", "move_head", "play_emotion"}}
            assert set(names) <= expected_motion[row["expect"]] and names
```

- `test_mute_blocks_and_state_reports` asserts `_names(resp) == ["idle_do_nothing"]` right after `/api/mute`. With the gestures block added, the first muted response carries the sleep cue: change it to `assert _names(resp) == ["play_emotion"]`.

- [ ] **Step 5: Commit**

```bash
git add familiar_brain/app.py tests/test_app_expression.py tests/test_app_sight.py tests/test_app_integration.py
git commit -m "feat(app): gestures, reactions and attention on every turn; sight continuations; thinking mode"
```

---

## Part F: Robot-side files that need no robot

### Task 12: Robot profile, llama-server mmproj, the smoke-test script

**Files:**
- Create: `robot_profile/familiar/profile.md`, `docs/smoke-test.md`
- Modify: `scripts/start_all.sh`, `docs/deviations.md`, `README.md`

Everything in this task is done at the desk. Running the smoke test is the deferred live check.

- [ ] **Step 1: Write `robot_profile/familiar/profile.md`** (app 1.0.1 profile schema; the older `instructions.md` and `tools.txt` stay until the live check confirms the installed app version)

```markdown
+++
schema_version = 1
voice = "Aiden"
hidden = false
default_tools = [
  "move_head",
  "play_emotion",
  "stop_emotion",
  "idle_do_nothing",
  "head_tracking",
  "camera",
]
+++

## Identity

Placeholder. The familiar brain injects the real system prompt on every turn.
```

`dance`, `sweep_look`, `remember`, `forget`, `go_to_sleep` and every Space tool stay off: a dance at the table is wrong, `remember` would store player facts on the robot, and `go_to_sleep` stops the app.

- [ ] **Step 2: Add the projector to `scripts/start_all.sh`**

llama-server already auto-downloads the mmproj for an `-hf` repo that has one (`--help`: "mmproj is also downloaded automatically if available"), so the current line may already be loading it. Make it explicit either way. Change the llama-server line to:

```bash
"$LLAMA_BIN" -hf "$MODEL" ${MMPROJ_ARGS:---mmproj-auto} -np 2 -c 32768 -fa on --swa-full --jinja --port 8080 --host 127.0.0.1 \
  > data/llama.log 2>&1 &
```

with, above it:

```bash
# Gemma 4 E4B ships mmproj-gemma-4-E4B-it-{BF16,Q8_0}.gguf in the same repo. The
# projector is what lets the camera tool's frame reach the model (Plan 3, sight).
# Set MMPROJ_ARGS=--no-mmproj to measure VRAM without it.
```

- [ ] **Step 3: Write `docs/smoke-test.md`**

```markdown
# Smoke test and probes for Plan 3

Run with the robot on, `./scripts/start_all.sh` up, and a session started. Record
every result in docs/versions.md under "Plan 3 probes". A probe that fails stops the
check at that point; do not work around it silently.

## Probes (in order)

P0. Tool-result echoes. Say a line that is not addressed to Rocky. In data/brain.log
    count the timing lines it produced. One line (decision IGNORE) means the app does
    not return idle_do_nothing's result; two (IGNORE then ECHO) means it does and the
    empty response closed it. Either is fine. A third line, or speech, is a defect.

P1. Gesture plus speech in one response. "Rocky, what do you make of the smell?"
    Expect: head turns to you, an expression plays, then Rocky speaks. brain.log shows
    "gesture mood=" and "head tracking on". If the robot plays the emotion but never
    speaks, speech-to-speech dropped the text after a tool call: record it and stop.

P2. Tracking hands back. Stay quiet for follow_up_window_s + tracking_linger_s, then
    say any table line. brain.log shows "head tracking off". Say a line from the left
    of the robot: expect "glance left" and a head turn. If the head does not move,
    check GET http://10.0.0.100:8000/api/state/doa by hand and record the payload.

P3. Narration reactions. Read three lines as the Dungeon Master, at least fifteen
    seconds apart, none naming Rocky:
      a. "The priest pulls back his hood. It is the king, and he has been dead a year."
         Expect the surprised move; brain.log shows "reaction call answered 'alarmed'".
      b. "She folds the letter and does not read the rest aloud."
         Expect the sad move.
      c. "You walk for an hour along the river road."
         Expect nothing; brain.log shows the reaction call answered 'none'.
    Then read a and b again ten seconds apart: the second one must produce no
    reaction (reaction_gap_s). Record the "preface" figure from the timing lines. If
    it is over 100 ms, or if c reacts, note it: D9's reversal applies.

P4. Sight. "Rocky, what do you see?" Expect the camera tool in brain.log's upstream
    request (a function_call named camera streamed to the client), then a SIGHT timing
    line, then a spoken description. data/llama.log must show the mmproj loaded and an
    image being encoded. Record VRAM from nvidia-smi with the frame in flight. If
    llama-server returns 4xx on the image request, record the body and stop: the
    reversal in D5 applies.

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
```

- [ ] **Step 4: Record deviations and the README paragraph**

Append a "Deviations from Plan 3" section to `docs/deviations.md` in the same style as the Plan 2 section: one entry per place the code as written differs from this plan, with the reason and the test that pins it. If nothing deviated, say so in one line under the heading. Add a short "Senses and expression" paragraph to `README.md` naming: the mmproj requirement, the DoA read, that gestures and reactions are chosen by the brain from keyword rules and a one-token call (never by the generating model), and that frames are never written to disk or the session log.

- [ ] **Step 5: Commit**

```bash
git add robot_profile/familiar/profile.md scripts/start_all.sh docs/smoke-test.md docs/deviations.md README.md
git commit -m "chore(robot): profile.md for app 1.0.1, explicit mmproj, Plan 3 smoke test and deviations"
```

---

## Deferred

These are not part of executing this plan. They are listed so nothing is lost.

**Live check (TJ, later session, robot on).** Follow `docs/smoke-test.md`.
1. On the robot dashboard (`http://10.0.0.100:8000`) note the conversation app version. If 1.0.0 or later, copy `robot_profile/familiar/profile.md` to the app's external profiles directory per its README, select the `familiar` profile, and delete `robot_profile/familiar/instructions.md` and `tools.txt` from the repo. If older, keep the two files, record the version in `docs/versions.md`, and map the gestures back to move ids per deviation C7 (decision D3 reversal).
2. Run probes P0 to P6 and the hardware checklist. Paste each result, the "preface" timing figure, the mmproj VRAM figure and the dice score into `docs/versions.md` under a "Plan 3 probes" heading.
3. Apply the D6 reversal to `persona.py` if P5 fails; apply the D9 reversal if P3 reacts to the travel line or costs over 100 ms.
4. Commit: `docs: Plan 3 probe results`.

**Deep night (needs a conversation with the Dungeon Master first).** The earlier draft of this plan had "deep night" from anyone put Rocky to sleep (gate closed, camera closed, sleepy gesture) and "deep night is over" or "good morning" wake it, as a flag separate from mute so the phrase could never undo a privacy mute. TJ removed it pending that conversation. If it comes back, it is: an `asleep` flag on `BrainState`, `sleep_transition(text)` phrase rules in `motion.py`, `muted or asleep` in `Gate.decide`, `Attention.wants_tracking` and the sight branch of the responses route, `asleep` in `/api/state`, and one probe. Nothing in the present plan needs to change to add it.

**Face identification (Session Zero consent).** Decision D7. If the players consent, Plan 4 scopes it. The present plan leaves no hook for it on purpose.

**Dice detector.** Decision D6 reversal, only if TJ wants dice sight regardless of the probe.

---

## Self-review

**Spec coverage.** Design 4.7 face tracking: Task 8 and 11. Attend routine: the robot's own idle policy plus DoA glances (D4 states the reduction). Camera as a sense with local description: Tasks 10, 11, 12 (mmproj). Reactive gestures on non-addressed lines, including narration: Task 3 (`for_table`, `TABLE_RULES`, `REACTION_MOODS`), Task 4 (`REACTION_INSTRUCTION`), Task 11 (`_preface`) and probe P3. Consent, mute closes camera: Task 11 sight branch and test. Design 5 decision effects, AMBIENT gesture by keyword: Task 3. Antenna flick before speech: the `attentive` gesture on ADDRESSED. Persona gestures from the app's library: Task 1. The request's own items: name Rocky (Task 1), expressive (Tasks 2 to 5, 11), reacts to the Dungeon Master's twists and sad moments (Tasks 3, 4, 11), face tracking (Task 8), sees the room and the dice (Tasks 10 to 12, D6), pet and companion (attention plus gestures plus the existing memory), thought partner (Tasks 6, 9, 11). Deep night: removed on TJ's instruction, recorded under "Deferred". Live check: deferred, recorded under "Deferred".

**Not covered, deliberately.** Face identification (D7). A dice detector (D6). A brain-side attend timer (D4). Companion UI buttons for mode: the endpoint exists; the UI is still design milestone 8, unbuilt. Telling the Dungeon Master's voice from a player's: the brain has no speaker identity, so reactions apply to any line not addressed to Rocky.

**Type consistency.** `call_item(name, args) -> dict`, `item_events(items, *, start_index)`, `tool_call_events(items, *, response_id)`, `empty_response_events`, `idle_items`, `continuation_kind(req) -> str | None` in Tasks 2, 10, 11. `Choreographer(persona, *, min_gap_s, reaction_gap_s, clock, mood_classifier, reaction_classifier)` with `for_addressed(text, now)`, `for_table(text, now)`, `cue(mood, now)` in Tasks 3 and 11. `REACTION_MOODS` in Tasks 3 and 11. `Attention(*, state, doa, follow_up_window_s, linger_s, glance_gap_s).items(decision, now)` in Tasks 8 and 11. `RobotDaemon(base_url, *, timeout_s, get).doa()` in Tasks 7 and 11. `update_mode(state, text, *, addressed, now, timeout_s)` in Tasks 9 and 11. `render_system_prompt(p, context, mode)` in Tasks 6 and 11. `BrainState.set_cue, consume_cue, tracking, last_glance_at, mode, mode_set_at` in Tasks 5, 8, 9, 11. `LLMClient.mood(text, *, moods, prompt)` and `REACTION_INSTRUCTION` in Tasks 4 and 11. `cfg.robot.daemon_url, doa_timeout_s`, `cfg.motion.min_gesture_gap_s, reaction_gap_s, glance_gap_s, tracking_linger_s, mode_timeout_s` in Tasks 5 and 11.
