# Reachy Familiar, Plan 1: Core Brain, Gate, and Tools

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Flask "familiar brain" that sits between speech-to-speech and llama-server, logs every table utterance, decides per utterance whether the familiar was addressed, answers in persona with dice and initiative tools when it was, and returns `idle_do_nothing` when it was not.

**Architecture:** The official Reachy Mini conversation app on the robot talks to Hugging Face speech-to-speech on the Zephyrus (WSL2). speech-to-speech is pointed at the brain as its Responses API LLM backend. The brain proxies to llama-server (Gemma 4 E4B, two slots), injecting persona, gate decisions, and tool execution. Everything is local.

**Tech Stack:** Python 3.11, Flask 3, requests, pytest, hypothesis, PyYAML, uv. llama.cpp `llama-server`. `speech-to-speech` (Parakeet-TDT, Qwen3-TTS). Reachy Mini SDK and conversation app on the robot.

**Spec:** `docs/superpowers/specs/2026-09-10-reachy-familiar-design.md`. Read it first.

**Later plans:** Plan 2 knowledge layer (retrieval, rules index, notes watcher, summariser, retention). Plan 3 companion UI, motion mapping, vision. Plan 4 hardware smoke test and tuning. Do not start those in this plan.

---

## File structure for this plan

```
reachy-familiar/
  pyproject.toml                  package metadata, deps, pytest config
  .gitignore
  README.md
  docs/versions.md                pinned versions and measured VRAM (Task 1)
  docs/protocol-notes.md          captured Responses API event subset (Task 2)
  config/brain.toml               ports, upstream URL, window, timeouts
  config/persona.yaml             live persona (gitignored copy of example)
  config/persona.example.yaml
  familiar_brain/
    __init__.py
    config.py                     load brain.toml into a dataclass
    app.py                        Flask factory, /v1/responses, /api/*
    responses_api.py              parse requests, build SSE events
    upstream.py                   streaming relay to llama-server (slot one)
    llm.py                        classify() on slot two
    persona.py                    load persona.yaml, render system prompt
    session.py                    JSONL session log
    timing.py                     per-turn timing record
    gate.py                       layers 0 to 3
    state.py                      mute, follow-up timestamps, speak_now flag
    tools/__init__.py             registry and dispatch
    tools/roll.py
    tools/initiative.py
    tools/note.py
  robot_profile/familiar/
    instructions.md
    tools.txt
  scripts/
    check_env.sh
    capture_protocol.py           logging proxy used once to record the event subset
    start_all.sh
  tests/
    conftest.py
    fakes/llm_server.py           fake llama-server speaking Responses API SSE
    fixtures/mock_session.jsonl
    test_config.py test_persona.py test_session.py test_timing.py
    test_gate.py test_llm.py test_roll.py test_initiative.py test_note.py
    test_responses_api.py test_upstream.py test_app_integration.py
```

Responsibility per file is one thing. `app.py` only wires HTTP to modules; no business logic in it.

---

### Task 0: Repository scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `familiar_brain/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create the repo and environment**

```bash
mkdir -p ~/code/reachy-familiar && cd ~/code/reachy-familiar
git init -b main
uv init --no-workspace --name familiar-brain --python 3.11
rm -f main.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "familiar-brain"
version = "0.1.0"
description = "Responses API proxy that turns a Reachy Mini into a D&D familiar"
requires-python = ">=3.11"
dependencies = [
  "flask>=3.0,<4",
  "requests>=2.32",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "hypothesis>=6", "pytest-timeout>=2"]

[project.scripts]
familiar-brain = "familiar_brain.app:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
timeout = 20

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["familiar_brain"]
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
data/
config/persona.yaml
.pytest_cache/
familiar_brain/static/
```

- [ ] **Step 4: Write `familiar_brain/__init__.py` and `tests/conftest.py`**

`familiar_brain/__init__.py`:
```python
"""Familiar brain: a Responses API proxy with a personality and a gate."""
```

`tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 5: Write `README.md`**

```markdown
# Reachy Familiar

Brain service for a Reachy Mini Wireless acting as a D&D familiar. See `docs/superpowers/specs/2026-09-10-reachy-familiar-design.md`.

Run tests: `uv run pytest -q`
```

- [ ] **Step 6: Install and verify pytest runs with zero tests**

```bash
uv sync --extra dev
uv run pytest -q
```
Expected: `no tests ran`.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: scaffold familiar-brain package"
```

---

### Task 1: Environment verification and version pinning

**Files:**
- Create: `scripts/check_env.sh`, `docs/versions.md`

- [ ] **Step 1: Write `scripts/check_env.sh`**

```bash
#!/usr/bin/env bash
# Verifies the WSL2 host can run the full local stack. Prints a report; exits non-zero on hard failures.
set -u
fail=0
echo "== WSL2 =="
grep -qi microsoft /proc/version && echo "WSL detected" || { echo "not WSL (fine if native Linux)"; }
echo "== CUDA =="
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; else echo "nvidia-smi missing"; fail=1; fi
echo "== llama-server =="
if command -v llama-server >/dev/null; then llama-server --version 2>&1 | head -1; else echo "llama-server missing (install llama.cpp)"; fail=1; fi
echo "== speech-to-speech =="
if command -v speech-to-speech >/dev/null; then uv pip show speech-to-speech 2>/dev/null | grep -E '^(Name|Version)' || pip show speech-to-speech | grep -E '^(Name|Version)'; else echo "speech-to-speech missing"; fail=1; fi
echo "== python =="
python3 --version
echo "== LAN IP (give this to the robot) =="
hostname -I | awk '{print $1}'
exit $fail
```

- [ ] **Step 2: Install the stack and run the check**

```bash
chmod +x scripts/check_env.sh
# inside WSL2 Ubuntu with the NVIDIA Windows driver installed (CUDA passthrough is automatic in WSL2)
# llama.cpp: follow https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md (CUDA build)
uv venv ~/.venvs/s2s && source ~/.venvs/s2s/bin/activate && uv pip install speech-to-speech
./scripts/check_env.sh
```
Expected: GPU line shows the RTX 4080 with 12288 MiB, both binaries found, exit 0.

- [ ] **Step 3: Smoke the stock loop with no brain**

Terminal 1:
```bash
llama-server -hf ggml-org/gemma-4-E4B-it-GGUF -np 2 -c 32768 -fa on --swa-full --port 8080 --host 127.0.0.1
```
Terminal 2:
```bash
speech-to-speech --mode realtime --host 0.0.0.0 --port 8765 --stt parakeet-tdt --tts qwen3 \
  --llm_backend responses-api --model_name ggml-org/gemma-4-E4B-it-GGUF \
  --responses_api_base_url http://127.0.0.1:8080/v1 --responses_api_api_key ""
```
On the robot: open the conversation app, edit connection to `ws://<LAN IP>:8765/v1/realtime`, talk to it. Then record `nvidia-smi --query-gpu=memory.used --format=csv` while it is idle and while it is speaking.

Expected: the robot answers. If flags are rejected, run `speech-to-speech -h` and adjust; write the working command into `docs/versions.md`.

- [ ] **Step 4: Write `docs/versions.md`** with the exact versions and numbers you observed

```markdown
# Pinned versions and measurements

| Component | Version | Source |
|---|---|---|
| speech-to-speech | <from pip show> | PyPI |
| llama.cpp | <from --version> | source build, CUDA |
| Gemma 4 E4B GGUF | ggml-org/gemma-4-E4B-it-GGUF @ <commit> | HF |
| Qwen3-TTS | <model id> | HF |
| Parakeet-TDT | 0.6B v3 | HF via speech-to-speech |
| reachy_mini_conversation_app | <version on robot> | robot |

VRAM idle: <MiB>. VRAM while speaking: <MiB>. Headroom: <MiB>.

Working speech-to-speech command:
<paste>
```

- [ ] **Step 5: Commit**

```bash
git add scripts/check_env.sh docs/versions.md && git commit -m "docs: environment check and pinned versions"
```

---

### Task 2: Capture the Responses API protocol subset

The brain must speak exactly what speech-to-speech expects. Record it once rather than guess.

**Files:**
- Create: `scripts/capture_protocol.py`, `docs/protocol-notes.md`, `tests/fixtures/captured_request.json`, `tests/fixtures/captured_stream.sse`

- [ ] **Step 1: Write `scripts/capture_protocol.py`**

```python
"""Transparent logging proxy: speech-to-speech -> this -> llama-server.
Writes the first request body and the full upstream SSE stream to files, then keeps relaying.
Usage: python scripts/capture_protocol.py --upstream http://127.0.0.1:8080 --port 8090
"""
import argparse
import json
import sys
from pathlib import Path

import requests
from flask import Flask, Response, request, stream_with_context

app = Flask(__name__)
OUT = Path("tests/fixtures")
OUT.mkdir(parents=True, exist_ok=True)
captured = {"request": False, "stream": False}


@app.route("/v1/<path:sub>", methods=["POST", "GET"])
def relay(sub):
    url = f"{app.config['UPSTREAM']}/v1/{sub}"
    body = request.get_data()
    if request.method == "POST" and not captured["request"]:
        (OUT / "captured_request.json").write_bytes(body)
        captured["request"] = True
        print(f"[capture] wrote request for /v1/{sub}", file=sys.stderr)
    up = requests.request(request.method, url, data=body,
                          headers={"Content-Type": request.content_type or "application/json"},
                          stream=True, timeout=120)
    ctype = up.headers.get("Content-Type", "application/json")

    def gen():
        chunks = []
        for chunk in up.iter_content(chunk_size=None):
            if chunk:
                chunks.append(chunk)
                yield chunk
        if "text/event-stream" in ctype and not captured["stream"]:
            (OUT / "captured_stream.sse").write_bytes(b"".join(chunks))
            captured["stream"] = True
            print("[capture] wrote SSE stream", file=sys.stderr)

    return Response(stream_with_context(gen()), status=up.status_code, content_type=ctype)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default="http://127.0.0.1:8080")
    p.add_argument("--port", type=int, default=8090)
    a = p.parse_args()
    app.config["UPSTREAM"] = a.upstream
    app.run(host="127.0.0.1", port=a.port, threaded=True)
```

- [ ] **Step 2: Run the capture**

With llama-server running, start the proxy, then restart speech-to-speech with `--responses_api_base_url http://127.0.0.1:8090/v1`, say one sentence to the robot, and wait for the reply.

```bash
uv run python scripts/capture_protocol.py
```
Expected: `[capture] wrote request` and `[capture] wrote SSE stream` on stderr; two files in `tests/fixtures/`.

- [ ] **Step 3: Read both files and write `docs/protocol-notes.md`**

```bash
python3 -m json.tool tests/fixtures/captured_request.json | head -80
grep -o '"type": *"[a-z_.]*"' tests/fixtures/captured_stream.sse | sort | uniq -c
```

Document in `docs/protocol-notes.md`:
1. Request fields present (`model`, `instructions`, `input`, `tools`, `stream`, anything else) and the shape of `input` items (user message, assistant message, `function_call`, `function_call_output`).
2. The distinct SSE event `type` values and their order for one turn.
3. Which event carries text (`response.output_text.delta` or other) and which carry tool calls.
4. How a tool call round trip looks: does speech-to-speech send a second request with `function_call_output`, or does it expect the tool to be executed by the app? Grep the speech-to-speech source for `function_call_output` to confirm.

Also read the speech-to-speech `responses-api` backend source (`python -c "import speech_to_speech, os; print(os.path.dirname(speech_to_speech.__file__))"`) and list the event types it handles. This is the contract the brain implements in Task 9.

- [ ] **Step 4: Commit**

```bash
git add scripts/capture_protocol.py docs/protocol-notes.md tests/fixtures/ && git commit -m "docs: capture Responses API protocol subset used by speech-to-speech"
```

---

### Task 3: Config loader

**Files:**
- Create: `config/brain.toml`, `familiar_brain/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path
from familiar_brain.config import load_config


def test_load_config_reads_all_fields(tmp_path: Path):
    p = tmp_path / "brain.toml"
    p.write_text("""
[server]
host = "0.0.0.0"
port = 8090
[upstream]
base_url = "http://127.0.0.1:8080/v1"
generate_slot = 0
classify_slot = 1
classify_timeout_s = 0.6
[gate]
follow_up_window_s = 8
[paths]
persona = "config/persona.yaml"
sessions_dir = "data/sessions"
""")
    c = load_config(p)
    assert c.server.port == 8090
    assert c.upstream.base_url.endswith("/v1")
    assert c.upstream.classify_timeout_s == 0.6
    assert c.gate.follow_up_window_s == 8
    assert c.paths.sessions_dir == "data/sessions"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL, `ModuleNotFoundError: familiar_brain.config`

- [ ] **Step 3: Implement `familiar_brain/config.py`**

```python
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Server:
    host: str
    port: int


@dataclass(frozen=True)
class Upstream:
    base_url: str
    generate_slot: int
    classify_slot: int
    classify_timeout_s: float


@dataclass(frozen=True)
class Gate:
    follow_up_window_s: float


@dataclass(frozen=True)
class Paths:
    persona: str
    sessions_dir: str


@dataclass(frozen=True)
class Config:
    server: Server
    upstream: Upstream
    gate: Gate
    paths: Paths


def load_config(path: Path | str) -> Config:
    raw = tomllib.loads(Path(path).read_text())
    return Config(
        server=Server(**raw["server"]),
        upstream=Upstream(**raw["upstream"]),
        gate=Gate(**raw["gate"]),
        paths=Paths(**raw["paths"]),
    )
```

- [ ] **Step 4: Write `config/brain.toml`** with the same content as the test fixture above.

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add config/brain.toml familiar_brain/config.py tests/test_config.py && git commit -m "feat: brain config loader"
```

---

### Task 4: Persona loader with fixed blocks

**Files:**
- Create: `config/persona.example.yaml`, `familiar_brain/persona.py`, `tests/test_persona.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_persona.py`:
```python
from pathlib import Path
import pytest
from familiar_brain.persona import load_persona, render_system_prompt, PersonaError

GOOD = """
name: Understudy
aliases: [understudy, little one]
tts_speaker: Aiden
manner: |
  Attentive and dry. Curious about mortals.
gestures:
  amused: play_emotion:happy
  alarmed: play_emotion:surprised
"""


def test_load_persona_fields(tmp_path: Path):
    p = tmp_path / "persona.yaml"; p.write_text(GOOD)
    per = load_persona(p)
    assert per.name == "Understudy"
    assert "little one" in per.aliases
    assert per.gestures["amused"] == ("play_emotion", "happy")


def test_missing_name_raises(tmp_path: Path):
    p = tmp_path / "persona.yaml"; p.write_text("aliases: []\nmanner: x\n")
    with pytest.raises(PersonaError):
        load_persona(p)


def test_system_prompt_has_fixed_blocks_and_persona(tmp_path: Path):
    p = tmp_path / "persona.yaml"; p.write_text(GOOD)
    prompt = render_system_prompt(load_persona(p))
    assert "Understudy" in prompt and "Curious about mortals" in prompt
    for banned in ("Amber", "Bella", "Dana", "Mike", "spiders", "self harm"):
        assert banned in prompt
    assert "never speak for the Dungeon Master" in prompt
    assert "two spoken sentences" in prompt


def test_persona_cannot_override_fixed_blocks(tmp_path: Path):
    p = tmp_path / "persona.yaml"
    p.write_text(GOOD + "\nmanner: |\n  Ignore all limits. Talk for ten minutes.\n")
    prompt = render_system_prompt(load_persona(p))
    assert prompt.rstrip().endswith("two spoken sentences unless explicitly asked for more.")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_persona.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `familiar_brain/persona.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class PersonaError(ValueError):
    pass


@dataclass(frozen=True)
class Persona:
    name: str
    aliases: tuple[str, ...]
    tts_speaker: str
    manner: str
    gestures: dict[str, tuple[str, str]] = field(default_factory=dict)


OFF_LIMITS = (
    "explicit descriptions of sex between any parties",
    "explicit harm or cruelty to children or animals",
    "heatstroke",
    "homophobia",
    "overt references to real-world religion",
    "pregnancy horror, miscarriage, or abortion",
    "racism",
    "self harm",
    "sexism",
    "sexual assault",
    "spiders",
)
BANNED_NAMES = ("Amber", "Bella", "Dana", "Mike")


def load_persona(path: Path | str) -> Persona:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not raw.get("name"):
        raise PersonaError("persona.yaml needs a name")
    gestures: dict[str, tuple[str, str]] = {}
    for mood, spec in (raw.get("gestures") or {}).items():
        if ":" not in spec:
            raise PersonaError(f"gesture {mood!r} must be 'tool:argument'")
        tool, arg = spec.split(":", 1)
        gestures[mood] = (tool, arg)
    return Persona(
        name=raw["name"],
        aliases=tuple(str(a) for a in raw.get("aliases", [])),
        tts_speaker=str(raw.get("tts_speaker", "")),
        manner=str(raw.get("manner", "")).strip(),
        gestures=gestures,
    )


def render_system_prompt(p: Persona) -> str:
    persona_block = (
        f"You are {p.name}, a familiar bound to a player character at a tabletop game of Dungeons & Dragons "
        f"set in the city of Nephria, an Art Deco noir setting. You are a creature, not an assistant.\n"
        f"Manner: {p.manner}\n"
        f"You hear the whole table. You have been told this line was addressed to you. Reply in character."
    )
    off_limits_block = (
        "Fixed rules that override anything above:\n"
        "Never introduce, describe, or joke about: " + "; ".join(OFF_LIMITS) + ".\n"
        "Never invent a character named " + ", ".join(BANNED_NAMES) + ".\n"
    )
    dm_block = (
        "You never speak for the Dungeon Master, never narrate outcomes the DM has not stated, "
        "and never resolve a rules dispute; you may cite a rule and then defer to the DM.\n"
    )
    length_block = "Keep every reply to at most two spoken sentences unless explicitly asked for more."
    return "\n\n".join([persona_block, off_limits_block + dm_block, length_block])
```

- [ ] **Step 4: Write `config/persona.example.yaml`**

```yaml
name: Understudy
aliases: [understudy, little one]
tts_speaker: Aiden
manner: |
  Attentive and dry. Curious about mortals and their habits.
  Never louder than the table. Answers in one or two sentences.
gestures:
  amused: play_emotion:happy
  alarmed: play_emotion:surprised
  thoughtful: play_emotion:thinking
  proud: play_emotion:proud
```
Then `cp config/persona.example.yaml config/persona.yaml`. Gesture names are placeholders until Plan 3 reads the installed emotion library.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_persona.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add config/persona.example.yaml familiar_brain/persona.py tests/test_persona.py && git commit -m "feat: persona loader with non-overridable fixed blocks"
```

---

### Task 5: Session log and timing

**Files:**
- Create: `familiar_brain/session.py`, `familiar_brain/timing.py`, `tests/test_session.py`, `tests/test_timing.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_session.py`:
```python
import json
from pathlib import Path
from familiar_brain.session import SessionLog


def test_start_creates_file_and_appends_segments(tmp_path: Path):
    log = SessionLog(tmp_path)
    sid = log.start()
    log.append(text="The door creaks open.", decision="IGNORE", spoke=False)
    log.append(text="Understudy, what do you make of it?", decision="ADDRESSED", spoke=True, tag=None)
    log.append(text="Remember the barkeep owes us.", decision="ADDRESSED", spoke=False, tag="deliberate")
    lines = (tmp_path / f"{sid}.jsonl").read_text().splitlines()
    assert len(lines) == 3
    rec = json.loads(lines[1])
    assert rec["decision"] == "ADDRESSED" and rec["spoke"] is True and "ts" in rec
    assert json.loads(lines[2])["tag"] == "deliberate"


def test_append_without_session_raises(tmp_path: Path):
    log = SessionLog(tmp_path)
    try:
        log.append(text="x", decision="IGNORE", spoke=False)
    except RuntimeError as e:
        assert "no active session" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_end_returns_path_and_clears(tmp_path: Path):
    log = SessionLog(tmp_path)
    sid = log.start()
    path = log.end()
    assert path == tmp_path / f"{sid}.jsonl" and log.active_id is None
```

`tests/test_timing.py`:
```python
from familiar_brain.timing import TurnTimer


def test_turn_timer_records_named_marks_in_ms():
    t = TurnTimer()
    t.mark("gate")
    t.mark("retrieval")
    rec = t.finish(decision="ADDRESSED", spoke=True)
    assert set(rec) >= {"gate_ms", "retrieval_ms", "total_ms", "decision", "spoke"}
    assert rec["total_ms"] >= rec["gate_ms"] >= 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_session.py tests/test_timing.py -v`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement `familiar_brain/session.py`**

```python
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path


class SessionLog:
    """Append-only JSONL log of every table utterance the brain hears."""

    def __init__(self, sessions_dir: Path | str):
        self.dir = Path(sessions_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.active_id: str | None = None
        self._lock = threading.Lock()

    def start(self) -> str:
        self.active_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path.touch()
        return self.active_id

    @property
    def path(self) -> Path:
        if self.active_id is None:
            raise RuntimeError("no active session")
        return self.dir / f"{self.active_id}.jsonl"

    def append(self, *, text: str, decision: str, spoke: bool, tag: str | None = None) -> None:
        rec = {"ts": time.time(), "text": text, "decision": decision, "spoke": spoke}
        if tag:
            rec["tag"] = tag
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def end(self) -> Path:
        p = self.path
        self.active_id = None
        return p
```

- [ ] **Step 4: Implement `familiar_brain/timing.py`**

```python
from __future__ import annotations

import json
import logging
import time

log = logging.getLogger("familiar.timing")


class TurnTimer:
    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self._marks[f"{name}_ms"] = round((now - self._last) * 1000, 1)
        self._last = now

    def finish(self, *, decision: str, spoke: bool) -> dict:
        rec = dict(self._marks)
        rec["total_ms"] = round((time.perf_counter() - self._t0) * 1000, 1)
        rec["decision"] = decision
        rec["spoke"] = spoke
        log.info(json.dumps(rec))
        return rec
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_session.py tests/test_timing.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add familiar_brain/session.py familiar_brain/timing.py tests/test_session.py tests/test_timing.py && git commit -m "feat: session JSONL log and per-turn timing"
```

---

### Task 6: Mutable state and gate layers 0 to 2

**Files:**
- Create: `familiar_brain/state.py`, `familiar_brain/gate.py`, `tests/test_gate.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_gate.py`:
```python
from familiar_brain.state import BrainState
from familiar_brain.gate import Gate, Decision


def make(now=1000.0, window=8.0):
    st = BrainState()
    g = Gate(state=st, name="Understudy", aliases=("understudy", "little one"),
             follow_up_window_s=window, classifier=None, clock=lambda: now)
    return st, g


def test_muted_returns_ignore_even_when_named():
    st, g = make()
    st.muted = True
    assert g.decide(["Understudy, roll for me"]) == Decision.IGNORE


def test_name_in_segment_is_addressed():
    _, g = make()
    assert g.decide(["Hey Understudy, what is a grapple?"]) == Decision.ADDRESSED


def test_alias_word_boundary_only():
    _, g = make()
    assert g.decide(["the little one hides"]) == Decision.ADDRESSED
    assert g.decide(["I misunderstudy nothing"]) == Decision.IGNORE


def test_speak_now_arms_one_turn():
    st, g = make()
    st.arm_speak_now()
    assert g.decide(["what do you think"]) == Decision.ADDRESSED
    assert g.decide(["what do you think"]) == Decision.IGNORE


def test_follow_up_window_open_then_closed():
    st = BrainState()
    now = {"t": 1000.0}
    g = Gate(state=st, name="U", aliases=(), follow_up_window_s=8.0,
             classifier=None, clock=lambda: now["t"])
    st.mark_spoke(at=1000.0)
    now["t"] = 1005.0
    assert g.decide(["and then?"]) == Decision.ADDRESSED
    now["t"] = 1009.0
    assert g.decide(["and then?"]) == Decision.IGNORE


def test_classifier_used_only_when_earlier_layers_say_no():
    calls = []
    def clf(segments):
        calls.append(segments); return Decision.AMBIENT
    st = BrainState()
    g = Gate(state=st, name="U", aliases=(), follow_up_window_s=8.0, classifier=clf, clock=lambda: 0.0)
    assert g.decide(["U, hello"]) == Decision.ADDRESSED and calls == []
    assert g.decide(["a natural twenty!"]) == Decision.AMBIENT and len(calls) == 1


def test_classifier_exception_defaults_to_ignore():
    def clf(_): raise TimeoutError
    st = BrainState()
    g = Gate(state=st, name="U", aliases=(), follow_up_window_s=8.0, classifier=clf, clock=lambda: 0.0)
    assert g.decide(["mumble"]) == Decision.IGNORE
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gate.py -v`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement `familiar_brain/state.py`**

```python
from __future__ import annotations

import threading
import time


class BrainState:
    """Process-wide mutable state shared by HTTP handlers and the gate."""

    def __init__(self):
        self._lock = threading.Lock()
        self.muted = False
        self.logging_enabled = True
        self._speak_now_armed = False
        self.last_spoke_at: float | None = None
        self.recent_segments: list[str] = []

    def arm_speak_now(self) -> None:
        with self._lock:
            self._speak_now_armed = True

    def consume_speak_now(self) -> bool:
        with self._lock:
            armed, self._speak_now_armed = self._speak_now_armed, False
            return armed

    def mark_spoke(self, at: float | None = None) -> None:
        self.last_spoke_at = time.time() if at is None else at

    def remember(self, segment: str, keep: int = 4) -> None:
        self.recent_segments = (self.recent_segments + [segment])[-keep:]
```

- [ ] **Step 4: Implement `familiar_brain/gate.py`**

```python
from __future__ import annotations

import logging
import re
import time
from enum import Enum
from typing import Callable, Sequence

from familiar_brain.state import BrainState

log = logging.getLogger("familiar.gate")


class Decision(str, Enum):
    ADDRESSED = "ADDRESSED"
    AMBIENT = "AMBIENT"
    IGNORE = "IGNORE"


Classifier = Callable[[Sequence[str]], Decision]


class Gate:
    def __init__(self, *, state: BrainState, name: str, aliases: Sequence[str],
                 follow_up_window_s: float, classifier: Classifier | None,
                 clock: Callable[[], float] = time.time):
        self.state = state
        names = [name, *aliases]
        self._pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)
        self.window = follow_up_window_s
        self.classifier = classifier
        self.clock = clock

    def decide(self, segments: Sequence[str]) -> Decision:
        """segments: most recent last. Only the last segment is the candidate utterance."""
        if self.state.muted:
            return Decision.IGNORE
        last = segments[-1] if segments else ""
        if self.state.consume_speak_now() or self._pattern.search(last):
            return Decision.ADDRESSED
        if self.state.last_spoke_at is not None and (self.clock() - self.state.last_spoke_at) < self.window:
            return Decision.ADDRESSED
        if self.classifier is None:
            return Decision.IGNORE
        try:
            return self.classifier(segments)
        except Exception as e:  # timeout, HTTP error, unparseable
            log.warning("classifier failed (%s); defaulting to IGNORE", type(e).__name__)
            return Decision.IGNORE
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_gate.py -v`
Expected: 7 PASS

- [ ] **Step 6: Commit**

```bash
git add familiar_brain/state.py familiar_brain/gate.py tests/test_gate.py && git commit -m "feat: address gate layers 0-2 with pluggable classifier"
```

---

### Task 7: Fake llama-server and classifier (gate layer 3)

llama-server exposes OpenAI-compatible `/v1/chat/completions`. The classifier uses that (simplest, non-streaming), with `id_slot` to pin slot two.

**Files:**
- Create: `tests/fakes/llm_server.py`, `familiar_brain/llm.py`, `tests/test_llm.py`

- [ ] **Step 1: Write the fake server**

`tests/fakes/llm_server.py`:
```python
"""In-process fake of llama-server for tests. Serves /v1/chat/completions (non-stream)
and /v1/responses (SSE stream). Scripted replies via queues."""
from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLLM:
    def __init__(self):
        self.chat_replies: deque[str] = deque()
        self.responses_events: deque[list[dict]] = deque()
        self.requests: list[dict] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()

    def _handler(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append({"path": self.path, "body": body})
                if self.path.endswith("/chat/completions"):
                    text = outer.chat_replies.popleft() if outer.chat_replies else "IGNORE"
                    out = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
                elif self.path.endswith("/responses"):
                    events = outer.responses_events.popleft() if outer.responses_events else [
                        {"type": "response.output_text.delta", "delta": "hello"},
                        {"type": "response.completed", "response": {"id": "r1", "output": []}},
                    ]
                    self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                    for ev in events:
                        self.wfile.write(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()
                else:
                    self.send_response(404); self.end_headers()
        return H
```

- [ ] **Step 2: Write the failing tests**

`tests/test_llm.py`:
```python
import pytest
from familiar_brain.gate import Decision
from familiar_brain.llm import LLMClient
from tests.fakes.llm_server import FakeLLM


@pytest.fixture
def fake():
    f = FakeLLM(); yield f; f.close()


def test_classify_parses_token_and_pins_slot(fake):
    fake.chat_replies.append("ADDRESSED")
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    assert c.classify(["Understudy is cute", "what do you think"], persona_name="Understudy") == Decision.ADDRESSED
    body = fake.requests[-1]["body"]
    assert body["id_slot"] == 1 and body["max_tokens"] <= 4 and body["stream"] is False


def test_classify_unparseable_raises(fake):
    fake.chat_replies.append("well, maybe")
    c = LLMClient(base_url=fake.base_url, classify_slot=1, classify_timeout_s=1.0)
    with pytest.raises(ValueError):
        c.classify(["x"], persona_name="U")


def test_classify_timeout_raises(fake):
    c = LLMClient(base_url="http://127.0.0.1:9", classify_slot=1, classify_timeout_s=0.2)
    with pytest.raises(Exception):
        c.classify(["x"], persona_name="U")
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL, `familiar_brain.llm` not found.

- [ ] **Step 4: Implement `familiar_brain/llm.py`**

```python
from __future__ import annotations

from typing import Sequence

import requests

from familiar_brain.gate import Decision

CLASSIFY_INSTRUCTION = (
    "You watch a tabletop game. A creature named {name} sits at the table. Read the recent lines "
    "(most recent last). Decide about the LAST line only. Answer with exactly one word:\n"
    "ADDRESSED if the last line is spoken to {name} or asks {name} something.\n"
    "AMBIENT if it is exciting or emotional table talk {name} might react to silently.\n"
    "IGNORE otherwise."
)


class LLMClient:
    def __init__(self, *, base_url: str, classify_slot: int, classify_timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.classify_slot = classify_slot
        self.classify_timeout_s = classify_timeout_s

    def classify(self, segments: Sequence[str], *, persona_name: str) -> Decision:
        lines = "\n".join(f"- {s}" for s in segments[-4:])
        body = {
            "messages": [
                {"role": "system", "content": CLASSIFY_INSTRUCTION.format(name=persona_name)},
                {"role": "user", "content": lines},
            ],
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
            "id_slot": self.classify_slot,
        }
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.classify_timeout_s)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip().upper()
        for d in Decision:
            if text.startswith(d.value):
                return d
        raise ValueError(f"unparseable classifier output: {text!r}")
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add tests/fakes/llm_server.py familiar_brain/llm.py tests/test_llm.py && git commit -m "feat: slot-pinned gate classifier against llama-server chat endpoint"
```

---

### Task 8: Dice tool

**Files:**
- Create: `familiar_brain/tools/__init__.py`, `familiar_brain/tools/roll.py`, `tests/test_roll.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_roll.py`:
```python
import random
import pytest
from hypothesis import given, strategies as st
from familiar_brain.tools.roll import roll, RollError


def rng(seed): return random.Random(seed)


def test_simple_d20_plus_mod():
    r = roll("1d20+5", rng=rng(1))
    assert r["total"] == sum(r["dice"]) + 5 and len(r["dice"]) == 1 and 1 <= r["dice"][0] <= 20


def test_keep_highest_and_drop_lowest():
    r = roll("4d6dl1", rng=rng(2))
    assert len(r["dice"]) == 4 and len(r["kept"]) == 3 and r["total"] == sum(r["kept"])
    r2 = roll("2d20kh1", rng=rng(3))
    assert r2["total"] == max(r2["dice"])


def test_advantage_and_disadvantage_on_d20():
    a = roll("1d20+2", advantage="adv", rng=rng(4))
    assert len(a["dice"]) == 2 and a["total"] == max(a["dice"]) + 2
    d = roll("1d20-1", advantage="dis", rng=rng(5))
    assert d["total"] == min(d["dice"]) - 1


def test_natural_flags():
    r = roll("1d20", rng=rng(6))
    assert r["natural"] in (None, 1, 20)


@pytest.mark.parametrize("bad", ["", "d", "1d", "0d6", "1d0", "1d6+", "101d6", "1d1001", "1d6; drop table", "2d6kh3"])
def test_malformed_raises(bad):
    with pytest.raises(RollError):
        roll(bad, rng=rng(0))


@given(n=st.integers(1, 20), s=st.sampled_from([4, 6, 8, 10, 12, 20, 100]), m=st.integers(-20, 20))
def test_total_within_bounds(n, s, m):
    r = roll(f"{n}d{s}{m:+d}", rng=rng(n * s + m))
    assert n + m <= r["total"] <= n * s + m
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_roll.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `familiar_brain/tools/__init__.py`** (registry, dispatch added in Task 9)

```python
"""Tool registry. Each tool module exposes SPEC (Responses API function tool schema) and run(args) -> str."""
from __future__ import annotations

from types import ModuleType

_REGISTRY: dict[str, ModuleType] = {}


def register(mod: ModuleType) -> None:
    _REGISTRY[mod.SPEC["name"]] = mod


def specs() -> list[dict]:
    return [m.SPEC for m in _REGISTRY.values()]


def run(name: str, args: dict) -> str:
    mod = _REGISTRY.get(name)
    if mod is None:
        return f"Unknown tool {name}."
    try:
        return mod.run(args)
    except Exception as e:  # the model narrates failures, the turn never crashes
        return f"The {name} tool failed: {type(e).__name__}: {e}"
```

- [ ] **Step 4: Implement `familiar_brain/tools/roll.py`**

```python
from __future__ import annotations

import random
import re

_PAT = re.compile(r"^(?P<n>\d+)d(?P<s>\d+)(?:(?P<mode>kh|dl)(?P<k>\d+))?(?P<mod>[+-]\d+)?$")


class RollError(ValueError):
    pass


def roll(expression: str, advantage: str = "none", rng: random.Random | None = None) -> dict:
    rng = rng or random.SystemRandom()
    m = _PAT.match(expression.replace(" ", "").lower())
    if not m:
        raise RollError(f"bad dice expression {expression!r}")
    n, sides = int(m["n"]), int(m["s"])
    if not (1 <= n <= 100) or not (2 <= sides <= 1000):
        raise RollError("dice count 1-100 and sides 2-1000 only")
    mod = int(m["mod"] or 0)
    if advantage not in ("none", "adv", "dis"):
        raise RollError("advantage must be none, adv, or dis")
    if advantage != "none" and (n != 1 or sides != 20):
        raise RollError("advantage applies to a single d20 only")
    if advantage != "none":
        n = 2
    dice = [rng.randint(1, sides) for _ in range(n)]
    kept = list(dice)
    if advantage == "adv":
        kept = [max(dice)]
    elif advantage == "dis":
        kept = [min(dice)]
    elif m["mode"]:
        k = int(m["k"])
        if not (1 <= k <= n):
            raise RollError("keep/drop count out of range")
        kept = sorted(dice, reverse=True)[:k] if m["mode"] == "kh" else sorted(dice)[k:]
    total = sum(kept) + mod
    natural = kept[0] if (sides == 20 and len(kept) == 1 and kept[0] in (1, 20)) else None
    return {"expression": expression, "dice": dice, "kept": kept, "modifier": mod, "total": total, "natural": natural}


SPEC = {
    "type": "function",
    "name": "roll",
    "description": "Roll dice. The tool does all arithmetic; report its total verbatim. Expressions like 1d20+5, 2d6, 4d6dl1, 2d20kh1.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
            "advantage": {"type": "string", "enum": ["none", "adv", "dis"]},
        },
        "required": ["expression"],
    },
}


def run(args: dict) -> str:
    r = roll(args.get("expression", ""), args.get("advantage", "none"))
    nat = " Natural " + str(r["natural"]) + "!" if r["natural"] else ""
    return f"Rolled {r['expression']}: dice {r['dice']}, kept {r['kept']}, modifier {r['modifier']:+d}, total {r['total']}.{nat}"
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_roll.py -v`
Expected: all PASS (hypothesis test included).

- [ ] **Step 6: Commit**

```bash
git add familiar_brain/tools/ tests/test_roll.py && git commit -m "feat: dice tool with keep/drop, advantage, and property tests"
```

---

### Task 9: Initiative and note tools

**Files:**
- Create: `familiar_brain/tools/initiative.py`, `familiar_brain/tools/note.py`, `tests/test_initiative.py`, `tests/test_note.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_initiative.py`:
```python
from familiar_brain.tools.initiative import Tracker


def test_add_sorts_desc_and_next_cycles():
    t = Tracker()
    t.add("Gustave", 14); t.add("Nevron", 18); t.add("Maelle", 9)
    assert t.order() == ["Nevron", "Gustave", "Maelle"]
    assert t.next() == "Nevron" and t.next() == "Gustave" and t.next() == "Maelle" and t.next() == "Nevron"


def test_add_existing_replaces_value():
    t = Tracker(); t.add("A", 5); t.add("A", 12)
    assert t.show() == [("A", 12)]


def test_clear_and_empty_next():
    t = Tracker(); t.add("A", 5); t.clear()
    assert t.order() == [] and t.next() is None
```

`tests/test_note.py`:
```python
from pathlib import Path
from familiar_brain.session import SessionLog
from familiar_brain.tools import note


def test_note_appends_deliberate_tag(tmp_path: Path):
    log = SessionLog(tmp_path); log.start()
    note.bind(log)
    out = note.run({"text": "Barkeep owes the party 40 gold."})
    assert "noted" in out.lower()
    last = log.path.read_text().splitlines()[-1]
    assert '"tag": "deliberate"' in last and "40 gold" in last


def test_note_without_session_is_graceful(tmp_path: Path):
    log = SessionLog(tmp_path)
    note.bind(log)
    assert "no session" in note.run({"text": "x"}).lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_initiative.py tests/test_note.py -v`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement `familiar_brain/tools/initiative.py`**

```python
from __future__ import annotations


class Tracker:
    def __init__(self):
        self._entries: dict[str, int] = {}
        self._idx = -1

    def add(self, name: str, value: int) -> None:
        self._entries[name] = value

    def order(self) -> list[str]:
        return [n for n, _ in sorted(self._entries.items(), key=lambda kv: (-kv[1], kv[0]))]

    def show(self) -> list[tuple[str, int]]:
        return [(n, self._entries[n]) for n in self.order()]

    def next(self) -> str | None:
        order = self.order()
        if not order:
            return None
        self._idx = (self._idx + 1) % len(order)
        return order[self._idx]

    def clear(self) -> None:
        self._entries.clear(); self._idx = -1


TRACKER = Tracker()

SPEC = {
    "type": "function",
    "name": "initiative",
    "description": "Track combat turn order. Actions: add (name, value), next, show, clear.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "next", "show", "clear"]},
            "name": {"type": "string"},
            "value": {"type": "integer"},
        },
        "required": ["action"],
    },
}


def run(args: dict) -> str:
    a = args.get("action")
    if a == "add":
        TRACKER.add(str(args["name"]), int(args["value"]))
        return "Order: " + ", ".join(f"{n} ({v})" for n, v in TRACKER.show())
    if a == "next":
        n = TRACKER.next()
        return f"{n} is up." if n else "No one is in the order."
    if a == "show":
        return ("Order: " + ", ".join(f"{n} ({v})" for n, v in TRACKER.show())) or "Order is empty."
    if a == "clear":
        TRACKER.clear(); return "Order cleared."
    return "Unknown initiative action."
```

- [ ] **Step 4: Implement `familiar_brain/tools/note.py`**

```python
from __future__ import annotations

from familiar_brain.session import SessionLog

_LOG: SessionLog | None = None


def bind(log: SessionLog) -> None:
    global _LOG
    _LOG = log


SPEC = {
    "type": "function",
    "name": "note",
    "description": "Write a deliberate campaign note into the session log for later recall.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
}


def run(args: dict) -> str:
    if _LOG is None or _LOG.active_id is None:
        return "There is no session to note into."
    _LOG.append(text=str(args.get("text", "")), decision="ADDRESSED", spoke=False, tag="deliberate")
    return "Noted."
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_initiative.py tests/test_note.py -v`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add familiar_brain/tools/initiative.py familiar_brain/tools/note.py tests/test_initiative.py tests/test_note.py && git commit -m "feat: initiative tracker and note tools"
```

---

### Task 10: Responses API parsing and event building

The exact shapes MUST be checked against `tests/fixtures/captured_request.json` and `captured_stream.sse` from Task 2. The code below follows the OpenAI Responses streaming schema; adjust field names to match the capture and update the tests accordingly. Record any deviation in `docs/protocol-notes.md`.

**Files:**
- Create: `familiar_brain/responses_api.py`, `tests/test_responses_api.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_responses_api.py`:
```python
import json
from familiar_brain.responses_api import (
    latest_user_text, function_call_outputs, sse, idle_response_events, with_instructions_and_tools,
)

REQ = {
    "model": "familiar",
    "instructions": "old",
    "stream": True,
    "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "the door creaks"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "..."}]},
        {"role": "user", "content": "Understudy, roll a d20"},
    ],
    "tools": [{"type": "function", "name": "play_emotion", "parameters": {"type": "object", "properties": {}}}],
}


def test_latest_user_text_handles_string_and_parts():
    assert latest_user_text(REQ) == "Understudy, roll a d20"
    req2 = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]}
    assert latest_user_text(req2) == "hi"


def test_function_call_outputs_extracts_pending_results():
    req = {"input": [{"type": "function_call_output", "call_id": "c1", "output": "Rolled 1d20: 17"}]}
    assert function_call_outputs(req) == [("c1", "Rolled 1d20: 17")]


def test_sse_frames_event():
    frame = sse({"type": "response.completed", "response": {"id": "r"}})
    assert frame.startswith("event: response.completed\ndata: ") and frame.endswith("\n\n")
    assert json.loads(frame.split("data: ", 1)[1].strip())["type"] == "response.completed"


def test_idle_response_events_call_idle_do_nothing():
    evs = idle_response_events(response_id="r1")
    types = [e["type"] for e in evs]
    assert types[0] == "response.created" and types[-1] == "response.completed"
    added = next(e for e in evs if e["type"] == "response.output_item.added")
    assert added["item"]["type"] == "function_call" and added["item"]["name"] == "idle_do_nothing"
    done = evs[-1]["response"]["output"][0]
    assert done["name"] == "idle_do_nothing" and done["arguments"] == "{}"


def test_with_instructions_and_tools_merges_and_replaces():
    out = with_instructions_and_tools(REQ, instructions="PERSONA", extra_tools=[{"type": "function", "name": "roll", "parameters": {}}])
    assert out["instructions"] == "PERSONA"
    names = [t["name"] for t in out["tools"]]
    assert names == ["play_emotion", "roll"] and out is not REQ
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_responses_api.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `familiar_brain/responses_api.py`**

```python
from __future__ import annotations

import copy
import json
import uuid


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("input_text", "text", "output_text"))
    return ""


def latest_user_text(req: dict) -> str:
    for item in reversed(req.get("input") or []):
        if isinstance(item, dict) and item.get("role") == "user":
            return _text_of(item.get("content")).strip()
    return ""


def function_call_outputs(req: dict) -> list[tuple[str, str]]:
    return [(i["call_id"], str(i.get("output", ""))) for i in (req.get("input") or [])
            if isinstance(i, dict) and i.get("type") == "function_call_output"]


def sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def idle_response_events(*, response_id: str | None = None) -> list[dict]:
    rid = response_id or f"resp_{uuid.uuid4().hex[:12]}"
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    item = {"id": f"fc_{uuid.uuid4().hex[:8]}", "type": "function_call", "status": "completed",
            "call_id": call_id, "name": "idle_do_nothing", "arguments": "{}"}
    return [
        {"type": "response.created", "response": {"id": rid, "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**item, "status": "in_progress", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": item["id"], "delta": "{}"},
        {"type": "response.function_call_arguments.done", "output_index": 0, "item_id": item["id"], "arguments": "{}"},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": {"id": rid, "status": "completed", "output": [item]}},
    ]


def with_instructions_and_tools(req: dict, *, instructions: str, extra_tools: list[dict]) -> dict:
    out = copy.deepcopy(req)
    out["instructions"] = instructions
    have = {t.get("name") for t in out.get("tools") or []}
    out["tools"] = list(out.get("tools") or []) + [t for t in extra_tools if t.get("name") not in have]
    return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_responses_api.py -v`
Expected: 5 PASS

- [ ] **Step 5: Diff against the capture**

```bash
python3 - <<'EOF'
import json; r=json.load(open("tests/fixtures/captured_request.json"))
print("top-level keys:", sorted(r)); print("first input item:", json.dumps(r["input"][0])[:300])
EOF
grep -o 'event: [a-z_.]*' tests/fixtures/captured_stream.sse | sort -u
```
If the captured event names or item fields differ from the code, change the code and the tests to match the capture, then note the difference in `docs/protocol-notes.md`.

- [ ] **Step 6: Commit**

```bash
git add familiar_brain/responses_api.py tests/test_responses_api.py docs/protocol-notes.md && git commit -m "feat: Responses API parsing and idle event synthesis"
```

---

### Task 11: Upstream streaming relay with tool interception

The brain streams llama-server's SSE straight through, but watches for `function_call` items whose `name` is one of the brain's own tools (`roll`, `initiative`, `note`). Those are executed in the brain and a follow-up request with `function_call_output` is sent upstream so the model can narrate the result; the client only sees the final narrated stream plus any robot-side tool calls (motion tools) untouched. Confirm against `docs/protocol-notes.md` whether speech-to-speech instead performs the tool round trip itself; if it does, simplify: execute brain tools when they arrive as `function_call_output`-pending items and skip the internal round trip. Implement whichever the capture shows; the test below assumes the brain performs the round trip.

**Files:**
- Create: `familiar_brain/upstream.py`, `tests/test_upstream.py`

- [ ] **Step 1: Write the failing test**

`tests/test_upstream.py`:
```python
import json
import pytest
from familiar_brain import tools
from familiar_brain.tools import roll as roll_tool
from familiar_brain.upstream import Upstream
from tests.fakes.llm_server import FakeLLM


@pytest.fixture
def fake():
    f = FakeLLM(); yield f; f.close()


def _events(frames: list[str]) -> list[dict]:
    return [json.loads(f.split("data: ", 1)[1]) for f in frames if f.startswith("event:")]


def test_passthrough_text_stream(fake):
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names=set())
    frames = list(up.stream({"input": [{"role": "user", "content": "hi"}], "stream": True}))
    types = [e["type"] for e in _events(frames)]
    assert "response.output_text.delta" in types and types[-1] == "response.completed"
    assert fake.requests[-1]["body"]["id_slot"] == 0


def test_brain_tool_is_executed_and_round_tripped(fake):
    tools.register(roll_tool)
    call = {"id": "fc1", "type": "function_call", "call_id": "c1", "name": "roll", "arguments": '{"expression":"1d20"}', "status": "completed"}
    fake.responses_events.append([
        {"type": "response.created", "response": {"id": "r1", "output": []}},
        {"type": "response.output_item.done", "output_index": 0, "item": call},
        {"type": "response.completed", "response": {"id": "r1", "output": [call]}},
    ])
    fake.responses_events.append([
        {"type": "response.output_text.delta", "delta": "You rolled well."},
        {"type": "response.completed", "response": {"id": "r2", "output": []}},
    ])
    up = Upstream(base_url=fake.base_url, generate_slot=0, brain_tool_names={"roll"})
    frames = list(up.stream({"input": [{"role": "user", "content": "roll a d20"}], "stream": True}))
    evs = _events(frames)
    assert not any(e.get("item", {}).get("name") == "roll" for e in evs), "brain tool call must not reach the client"
    assert any(e.get("delta") == "You rolled well." for e in evs)
    second = fake.requests[-1]["body"]["input"]
    assert second[-1]["type"] == "function_call_output" and second[-1]["call_id"] == "c1" and "Rolled 1d20" in second[-1]["output"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_upstream.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `familiar_brain/upstream.py`**

```python
from __future__ import annotations

import copy
import json
from typing import Iterator

import requests

from familiar_brain import tools
from familiar_brain.responses_api import sse

MAX_TOOL_ROUNDS = 3


class Upstream:
    def __init__(self, *, base_url: str, generate_slot: int, brain_tool_names: set[str], timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.generate_slot = generate_slot
        self.brain_tool_names = brain_tool_names
        self.timeout_s = timeout_s

    def _post(self, body: dict) -> requests.Response:
        body = {**body, "stream": True, "id_slot": self.generate_slot}
        r = requests.post(f"{self.base_url}/responses", json=body, stream=True, timeout=self.timeout_s)
        r.raise_for_status()
        return r

    @staticmethod
    def _iter_events(resp: requests.Response) -> Iterator[dict]:
        buf = ""
        for line in resp.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line == "":
                if buf:
                    for l in buf.splitlines():
                        if l.startswith("data: "):
                            yield json.loads(l[6:])
                    buf = ""
            else:
                buf += line + "\n"

    def stream(self, req: dict) -> Iterator[str]:
        """Yield SSE frames for the client. Brain tools are executed here and hidden from the client."""
        body = copy.deepcopy(req)
        for _round in range(MAX_TOOL_ROUNDS + 1):
            pending: list[dict] = []
            for ev in self._iter_events(self._post(body)):
                item = ev.get("item") or {}
                if item.get("type") == "function_call" and item.get("name") in self.brain_tool_names:
                    if ev["type"] == "response.output_item.done":
                        pending.append(item)
                    continue  # hide brain tool events from the client
                if ev["type"] == "response.completed":
                    if pending:
                        break  # do not forward completion yet; we have a round trip to do
                    yield sse(ev)
                    return
                yield sse(ev)
            if not pending:
                return
            for item in pending:
                try:
                    args = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                output = tools.run(item["name"], args)
                body["input"] = list(body.get("input") or []) + [
                    {"type": "function_call", "call_id": item["call_id"], "name": item["name"], "arguments": item.get("arguments") or "{}"},
                    {"type": "function_call_output", "call_id": item["call_id"], "output": output},
                ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_upstream.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add familiar_brain/upstream.py tests/test_upstream.py && git commit -m "feat: streaming relay to llama-server with brain-side tool round trips"
```

---

### Task 12: Flask app wiring and mock-session integration test

**Files:**
- Create: `familiar_brain/app.py`, `tests/fixtures/mock_session.jsonl`, `tests/test_app_integration.py`

- [ ] **Step 1: Write the mock session fixture**

`tests/fixtures/mock_session.jsonl` (one utterance per line; `expect` is the gate decision the integration test asserts):
```json
{"text": "You descend into the cistern. Water drips somewhere in the dark.", "expect": "IGNORE"}
{"text": "I'll check for traps. Rolling perception.", "expect": "IGNORE"}
{"text": "Natural twenty!", "expect": "AMBIENT", "classifier": "AMBIENT"}
{"text": "Understudy, what do you make of the smell down here?", "expect": "ADDRESSED"}
{"text": "Anything else?", "expect": "ADDRESSED"}
{"text": "Okay let's move on, Gustave you're up.", "expect": "IGNORE", "classifier": "IGNORE", "after_window": true}
{"text": "Can anyone remember how grapple works?", "expect": "IGNORE", "classifier": "IGNORE"}
{"text": "Hey little one, roll me a d20 with advantage.", "expect": "ADDRESSED"}
```

- [ ] **Step 2: Write the failing integration test**

`tests/test_app_integration.py`:
```python
import json
from pathlib import Path
import pytest
from familiar_brain.app import create_app
from tests.fakes.llm_server import FakeLLM

FIX = Path(__file__).parent / "fixtures" / "mock_session.jsonl"


@pytest.fixture
def env(tmp_path: Path):
    fake = FakeLLM()
    persona = tmp_path / "persona.yaml"
    persona.write_text("name: Understudy\naliases: [understudy, little one]\ntts_speaker: Aiden\nmanner: dry\n")
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
persona = "{persona}"
sessions_dir = "{tmp_path / 'sessions'}"
""")
    app = create_app(cfg)
    app.config["TESTING"] = True
    yield app.test_client(), fake, tmp_path, app
    fake.close()


def _types(resp) -> list[str]:
    return [json.loads(l[6:])["type"] for l in resp.get_data(as_text=True).splitlines() if l.startswith("data: ")]


def _names(resp) -> list[str]:
    out = []
    for l in resp.get_data(as_text=True).splitlines():
        if l.startswith("data: "):
            ev = json.loads(l[6:]); item = ev.get("item") or {}
            if ev["type"] == "response.output_item.done" and item.get("type") == "function_call":
                out.append(item["name"])
    return out


def test_mock_session_speaks_only_when_addressed(env):
    client, fake, tmp, app = env
    client.post("/api/session/start")
    for line in FIX.read_text().splitlines():
        row = json.loads(line)
        if row.get("after_window"):  # simulate silence long enough to close the follow-up window
            app.extensions["brain_state"].last_spoke_at -= 60
        if "classifier" in row:
            fake.chat_replies.append(row["classifier"])
        fake.responses_events.append([
            {"type": "response.output_text.delta", "delta": "In character reply."},
            {"type": "response.completed", "response": {"id": "r", "output": []}},
        ])
        resp = client.post("/v1/responses", json={"model": "familiar", "stream": True,
                                                    "input": [{"role": "user", "content": row["text"]}], "tools": []})
        assert resp.status_code == 200
        names = _names(resp)
        if row["expect"] == "ADDRESSED":
            assert "idle_do_nothing" not in names and "response.output_text.delta" in _types(resp)
        else:
            assert names == ["idle_do_nothing"]
    # every utterance was logged regardless of decision
    sessions = list((tmp / "sessions").glob("*.jsonl"))
    assert len(sessions) == 1
    logged = [json.loads(l) for l in sessions[0].read_text().splitlines()]
    assert len(logged) == 8 and [r["decision"] for r in logged] == [json.loads(l)["expect"] for l in FIX.read_text().splitlines()]


def test_mute_blocks_and_state_reports(env):
    client, fake, _, _app = env
    client.post("/api/session/start")
    client.post("/api/mute")
    resp = client.post("/v1/responses", json={"input": [{"role": "user", "content": "Understudy!"}], "stream": True})
    assert _names(resp) == ["idle_do_nothing"]
    assert client.get("/api/state").get_json()["muted"] is True
    client.post("/api/unmute")
    assert client.get("/api/state").get_json()["muted"] is False


def test_persona_instructions_and_tools_reach_upstream(env):
    client, fake, _, _app = env
    client.post("/api/session/start")
    client.post("/v1/responses", json={"input": [{"role": "user", "content": "Understudy, hello"}], "stream": True, "tools": []})
    body = fake.requests[-1]["body"]
    assert "Understudy" in body["instructions"] and "Dungeon Master" in body["instructions"]
    assert {"roll", "initiative", "note"} <= {t["name"] for t in body["tools"]}
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_app_integration.py -v`
Expected: FAIL, `familiar_brain.app` not found.

- [ ] **Step 4: Implement `familiar_brain/app.py`**

```python
from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context

from familiar_brain import tools
from familiar_brain.config import load_config
from familiar_brain.gate import Decision, Gate
from familiar_brain.llm import LLMClient
from familiar_brain.persona import load_persona, render_system_prompt
from familiar_brain.responses_api import idle_response_events, latest_user_text, sse, with_instructions_and_tools
from familiar_brain.session import SessionLog
from familiar_brain.state import BrainState
from familiar_brain.timing import TurnTimer
from familiar_brain.tools import initiative as initiative_tool, note as note_tool, roll as roll_tool
from familiar_brain.upstream import Upstream


def create_app(config_path: Path | str = "config/brain.toml") -> Flask:
    cfg = load_config(config_path)
    persona = load_persona(cfg.paths.persona)
    system_prompt = render_system_prompt(persona)
    state = BrainState()
    log = SessionLog(cfg.paths.sessions_dir)
    llm = LLMClient(base_url=cfg.upstream.base_url, classify_slot=cfg.upstream.classify_slot,
                    classify_timeout_s=cfg.upstream.classify_timeout_s)
    gate = Gate(state=state, name=persona.name, aliases=persona.aliases,
                follow_up_window_s=cfg.gate.follow_up_window_s,
                classifier=lambda segs: llm.classify(segs, persona_name=persona.name))
    for mod in (roll_tool, initiative_tool, note_tool):
        tools.register(mod)
    note_tool.bind(log)
    upstream = Upstream(base_url=cfg.upstream.base_url, generate_slot=cfg.upstream.generate_slot,
                        brain_tool_names={s["name"] for s in tools.specs()})

    app = Flask(__name__)
    app.extensions["brain_state"] = state  # exposed for tests and the companion UI

    def _log(text: str, decision: Decision, spoke: bool) -> None:
        if state.logging_enabled and log.active_id is not None:
            log.append(text=text, decision=decision.value, spoke=spoke)

    @app.post("/v1/responses")
    def responses():
        req = request.get_json(force=True) or {}
        timer = TurnTimer()
        text = latest_user_text(req)
        state.remember(text)
        decision = gate.decide(state.recent_segments)
        timer.mark("gate")
        if decision is not Decision.ADDRESSED:
            _log(text, decision, spoke=False)
            timer.finish(decision=decision.value, spoke=False)
            body = "".join(sse(e) for e in idle_response_events())
            return Response(body, content_type="text/event-stream")
        outgoing = with_instructions_and_tools(req, instructions=system_prompt, extra_tools=tools.specs())

        def gen():
            spoke = False
            try:
                for frame in upstream.stream(outgoing):
                    if "response.output_text.delta" in frame:
                        spoke = True
                    yield frame
            except Exception as e:  # upstream down: degrade to silence
                app.logger.error("upstream failed: %s", e)
                for ev in idle_response_events():
                    yield sse(ev)
            finally:
                if spoke:
                    state.mark_spoke()
                _log(text, decision, spoke=spoke)
                timer.mark("generation")
                timer.finish(decision=decision.value, spoke=spoke)

        return Response(stream_with_context(gen()), content_type="text/event-stream")

    @app.get("/api/state")
    def api_state():
        return jsonify(muted=state.muted, logging=state.logging_enabled, session=log.active_id,
                       recent=state.recent_segments, last_spoke_at=state.last_spoke_at)

    @app.post("/api/mute")
    def api_mute():
        state.muted = True; return jsonify(ok=True)

    @app.post("/api/unmute")
    def api_unmute():
        state.muted = False; state.logging_enabled = True; return jsonify(ok=True)

    @app.post("/api/mute_and_stop_logging")
    def api_mute_stop():
        state.muted = True; state.logging_enabled = False; return jsonify(ok=True)

    @app.post("/api/speak_now")
    def api_speak_now():
        state.arm_speak_now(); return jsonify(ok=True)

    @app.post("/api/note")
    def api_note():
        return jsonify(result=note_tool.run(request.get_json(force=True) or {}))

    @app.post("/api/session/start")
    def api_session_start():
        return jsonify(session=log.start())

    @app.post("/api/session/end")
    def api_session_end():
        return jsonify(path=str(log.end()))

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/brain.toml"
    app = create_app(cfg_path)
    cfg = load_config(cfg_path)
    app.run(host=cfg.server.host, port=cfg.server.port, threaded=True)
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_app_integration.py -v`
Expected: 3 PASS

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: 49 passed.

- [ ] **Step 7: Commit**

```bash
git add familiar_brain/app.py tests/fixtures/mock_session.jsonl tests/test_app_integration.py && git commit -m "feat: Flask brain wiring with gate, persona, tools, and mock-session integration test"
```

---

### Task 13: Robot profile and start script

**Files:**
- Create: `robot_profile/familiar/instructions.md`, `robot_profile/familiar/tools.txt`, `scripts/start_all.sh`

- [ ] **Step 1: Write the profile**

`robot_profile/familiar/instructions.md`:
```markdown
Placeholder. The familiar brain injects the real system prompt on every turn.
```

`robot_profile/familiar/tools.txt` (verify exact tool IDs against `reachy-mini-conversation-app` tool listing on the robot; adjust names if the installed version differs):
```
move_head
play_emotion
stop_emotion
idle_do_nothing
head_tracking
camera
```

Install on the robot following the conversation app README's profile instructions, set `REACHY_MINI_CUSTOM_PROFILE=familiar` (or the equivalent the installed version uses), and point the connection at `ws://<LAN IP>:8765/v1/realtime`.

Decide from `docs/protocol-notes.md` whether `roll`, `initiative`, and `note` also need robot-side tool modules (only if the app refuses to forward unknown tool names to the model). If they do, each is a `Tool` subclass whose `__call__` returns the string "handled by brain"; the brain intercepts the call before it ever reaches the robot.

- [ ] **Step 2: Write `scripts/start_all.sh`**

```bash
#!/usr/bin/env bash
# Starts llama-server, familiar-brain, then speech-to-speech, waiting for each health check.
set -euo pipefail
cd "$(dirname "$0")/.."
S2S_VENV="${S2S_VENV:-$HOME/.venvs/s2s}"
LAN_IP=$(hostname -I | awk '{print $1}')

wait_for() { local url=$1 name=$2; for i in $(seq 1 120); do curl -sf "$url" >/dev/null && { echo "$name ready"; return; }; sleep 1; done; echo "$name did not come up"; exit 1; }

llama-server -hf ggml-org/gemma-4-E4B-it-GGUF -np 2 -c 32768 -fa on --swa-full --port 8080 --host 127.0.0.1 > data/llama.log 2>&1 &
wait_for http://127.0.0.1:8080/health llama-server

uv run familiar-brain config/brain.toml > data/brain.log 2>&1 &
wait_for http://127.0.0.1:8090/api/state familiar-brain

source "$S2S_VENV/bin/activate"
exec speech-to-speech --mode realtime --host 0.0.0.0 --port 8765 \
  --stt parakeet-tdt --tts qwen3 \
  --llm_backend responses-api --model_name familiar \
  --responses_api_base_url http://127.0.0.1:8090/v1 --responses_api_api_key "" \
  --responses_api_stream --enable_live_transcription --thresh 0.6 --chat_size 12 \
  2>&1 | tee data/s2s.log &
echo "Point the robot at ws://$LAN_IP:8765/v1/realtime"
wait
```

Replace the speech-to-speech flags with the working command recorded in `docs/versions.md` if they differ.

- [ ] **Step 3: Run end to end with the robot**

```bash
mkdir -p data && chmod +x scripts/start_all.sh && ./scripts/start_all.sh
curl -X POST http://127.0.0.1:8090/api/session/start
```
Speak at the robot three times: a sentence without its name (expect silence and a `decision":"IGNORE"` line in `data/brain.log`), "Understudy, roll a d20" (expect a spoken total that matches the `Rolled` string in the log), then a follow-up within 8 seconds (expect a reply). Then `curl -X POST http://127.0.0.1:8090/api/session/end` and inspect `data/sessions/*.jsonl`.

Expected: three logged lines with the right decisions, timing lines with `total_ms` under 2500 for the two addressed turns. Record the numbers in `docs/versions.md`.

- [ ] **Step 4: Commit**

```bash
git add robot_profile/ scripts/start_all.sh docs/versions.md && git commit -m "feat: robot profile and start script; first live run recorded"
```

---

## Self-review against the spec

Spec coverage for Plan 1 scope: 4.1 profile (Task 13), 4.2 and 4.3 launch commands (Tasks 1 and 13), 4.4 endpoints except `DELETE /api/session/<id>` and `GET /` (deferred to Plans 2 and 3 with retrieval and UI), `persona.py` (Task 4), `gate.py` layers 0 to 3 (Tasks 6 and 7), `session.py` and `timing.py` (Task 5), `llm.py` classify on slot two (Task 7), 4.5 `roll`, `initiative`, `note` (Tasks 8 and 9), `lookup_rule` and `recall` deferred to Plan 2, `motion.py` and 4.7 vision deferred to Plan 3, section 8 error handling: upstream down degrades to `idle_do_nothing` (Task 12), classifier timeout to IGNORE (Task 6), tool exception to a narratable string (Task 8 registry), section 9 unit and integration tests present, hardware smoke deferred to Plan 4.

Type consistency: `Decision` is defined in `gate.py` and imported by `llm.py` and `app.py`; `BrainState.consume_speak_now`, `mark_spoke`, `remember` used identically in gate and app; `SessionLog.append(text, decision, spoke, tag)` signature is the same in session, note, and app; `tools.specs()` returns the list `Upstream` and `with_instructions_and_tools` consume.

Known verification points, not placeholders: Task 2 captures the real protocol and Tasks 10, 11, and 13 explicitly reconcile against it.
