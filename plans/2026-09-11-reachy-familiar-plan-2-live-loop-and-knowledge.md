# Reachy Familiar, Plan 2: Live Loop and Knowledge Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the address gate against real room audio with a scripted live session, then give the familiar memory: rules from both SRDs, campaign notes, and fiction-only summaries of past sessions, all retrieved locally on ADDRESSED turns.

**Architecture:** Part A changes the gate's follow-up anchoring and adds two CLIs for labelling and scoring a real session. Part B adds a SQLite plus sqlite-vec store with CPU embeddings, chunkers, an indexer CLI, two retrieval tools, a post-session summariser, and retention. Retrieval is injected into the upstream request only when the gate says ADDRESSED.

**Tech Stack:** Existing Plan 1 stack plus `sqlite-vec`, `sentence-transformers` (CPU), `pymupdf`. Rules PDFs: SRD 5.1 and SRD 5.2.1 from https://www.dndbeyond.com/srd (CC-BY-4.0).

**Spec:** `docs/superpowers/specs/2026-09-11-reachy-familiar-plan-2-spec.md`. Read it and the Plan 1 design first.

**Ground rule for this plan:** Plan 1 was modified during execution (see `docs/deviations.md`). Before editing any existing file, read it as it is now; the snippets below name functions, not line numbers, and describe the change. If a snippet conflicts with the current code, the current code wins on structure and this plan wins on behaviour. Write the behaviour test first either way.

**Gate:** Part B tasks (5 onward) do not start until Task 4's live session has been run and recorded. The summariser is tested against that real transcript, not a synthetic one.

---

## File structure for this plan

```
familiar_brain/
  state.py               modify: last_named_at, follow_up_grants
  gate.py                modify: anchoring in decide()
  cli.py                 create: label, gate-report, index subcommands
  embeddings.py          create: Embedder protocol, BgeEmbedder (CPU)
  retrieval.py           create: Store over sqlite + sqlite-vec
  chunking.py            create: pdf_chunks_by_bookmarks, markdown_chunks
  knowledge.py           create: build "What you know" block from Store
  summariser.py          create: fixed prompt, background job, retention
  tools/lookup_rule.py   create
  tools/recall.py        create
  app.py                 modify: knowledge injection, /api/reindex, /api/sessions, DELETE /api/session/<id>, retention at start
  config.py              modify: [knowledge] section
config/brain.toml        modify: [knowledge]
docs/live-loop.md        create
README.md                modify: SRD attribution
tests/
  test_gate_anchoring.py test_cli_label_report.py test_embeddings.py test_retrieval.py
  test_chunking.py test_knowledge.py test_lookup_recall.py test_summariser.py
  test_app_knowledge.py
  fakes/embedder.py
```

---

## Part A: Live loop

### Task 1: Re-anchor the follow-up window

**Files:**
- Modify: `familiar_brain/state.py`, `familiar_brain/gate.py`
- Test: `tests/test_gate_anchoring.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_gate_anchoring.py`:
```python
from familiar_brain.gate import Decision, Gate
from familiar_brain.state import BrainState


def make(clock):
    st = BrainState()
    g = Gate(state=st, name="Understudy", aliases=("little one",), follow_up_window_s=8.0,
             classifier=lambda segs: Decision.IGNORE, clock=clock)
    return st, g


def test_reply_to_reply_does_not_re_arm_window():
    """Reproduces the Plan 1 failure: six consecutive grants with nobody naming it."""
    now = {"t": 1000.0}
    st, g = make(lambda: now["t"])
    assert g.decide(["Understudy, hello"]) == Decision.ADDRESSED
    st.mark_spoke()                       # familiar replies at t=1000
    now["t"] += 3
    assert g.decide(["and then?"]) == Decision.ADDRESSED   # one follow-up is fine
    st.mark_spoke()                       # reply at t=1003
    now["t"] += 3                          # t=1006, still inside 8 s of last speech
    assert g.decide(["okay moving on"]) == Decision.ADDRESSED  # still inside the single extension
    st.mark_spoke()
    now["t"] += 3                          # t=1009: past naming (1000) + window (8) + one extension (8)? no: extension is from first reply
    # window = last_named_at + 8 = 1008, extended once to first_reply_at + 8 = 1008. t=1009 is outside.
    assert g.decide(["someone else talking"]) == Decision.IGNORE
    st.mark_spoke()                       # even if it somehow spoke, no re-arm
    now["t"] += 1
    assert g.decide(["more table talk"]) == Decision.IGNORE


def test_naming_again_reopens_window():
    now = {"t": 0.0}
    st, g = make(lambda: now["t"])
    g.decide(["Understudy?"]); st.mark_spoke()
    now["t"] = 100.0
    assert g.decide(["what now"]) == Decision.IGNORE
    assert g.decide(["little one, again"]) == Decision.ADDRESSED
    now["t"] = 104.0
    assert g.decide(["and?"]) == Decision.ADDRESSED


def test_speak_now_counts_as_naming():
    now = {"t": 0.0}
    st, g = make(lambda: now["t"])
    st.arm_speak_now()
    assert g.decide(["what do you think"]) == Decision.ADDRESSED
    now["t"] = 4.0
    assert g.decide(["really?"]) == Decision.ADDRESSED


def test_max_follow_ups_guard_still_applies():
    now = {"t": 0.0}
    st, g = make(lambda: now["t"])
    g.decide(["Understudy"]); 
    for i in range(3):
        now["t"] += 0.5
        assert g.decide([f"follow {i}"]) == Decision.ADDRESSED
    now["t"] += 0.5
    assert g.decide(["fourth"]) == Decision.IGNORE
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gate_anchoring.py -v`
Expected: the first test fails at the `IGNORE` assertion (current code keeps granting), others may fail on missing attributes.

- [ ] **Step 3: Modify `BrainState`** (read the current file first; add these members, keep everything else)

```python
        self.last_named_at: float | None = None      # set when name/alias or Speak Now opens the gate
        self.window_extended = False                # first reply after naming may extend once
        self.follow_up_grants = 0                   # existing Plan 1 counter, reset on naming

    def mark_named(self, at: float) -> None:
        self.last_named_at = at
        self.window_extended = False
        self.follow_up_grants = 0
```

`mark_spoke` stays as it is (records `last_spoke_at`); the gate now decides whether a speech extends anything.

- [ ] **Step 4: Modify `Gate.decide`** so layers 1 and 2 read:

```python
        now = self.clock()
        if self.state.consume_speak_now() or self._pattern.search(last):
            self.state.mark_named(now)
            return Decision.ADDRESSED
        if self.state.last_named_at is not None and self.state.follow_up_grants < self.max_follow_ups:
            deadline = self.state.last_named_at + self.window
            spoke = self.state.last_spoke_at
            if (not self.state.window_extended and spoke is not None
                    and self.state.last_named_at <= spoke <= deadline):
                deadline = spoke + self.window          # single extension from the first reply
                self.state.window_extended = True
            if now < deadline:
                self.state.follow_up_grants += 1
                return Decision.ADDRESSED
```

Keep `max_follow_ups` as the constructor argument Plan 1 added (default 3). Delete the old `last_spoke_at`-based branch so there is exactly one follow-up rule.

- [ ] **Step 5: Run the anchoring tests and the whole suite**

Run: `uv run pytest tests/test_gate_anchoring.py tests/test_gate.py -v && uv run pytest -q`
Expected: the four anchoring tests pass. Four Plan 1 tests in `tests/test_gate.py` fail for a correct reason: they name the familiar and then expect IGNORE on the next line while the clock is frozen, and naming now opens the window. Verified on a scratch copy of the Plan 1 code: `test_alias_word_boundary_only`, `test_speak_now_arms_one_turn`, `test_follow_up_window_open_then_closed`, `test_classifier_used_only_when_earlier_layers_say_no`. Fix each by giving the test a mutable clock and advancing it past the window (for example `now["t"] += 20`) between the naming line and the line expected to be IGNORE; for `test_follow_up_window_open_then_closed`, open the window with a naming line instead of `mark_spoke` alone, then keep its timing assertions. Do not weaken the assertions. One more: `tests/test_app_integration.py` advances only `last_spoke_at` for rows marked `after_window`; it must now also advance `last_named_at` by the same amount (verified on the scratch copy: this is the fifth and last Plan 1 test the anchoring change touches). Everything else green.

- [ ] **Step 6: Record the deviation and commit**

Append to `docs/deviations.md`: "Follow-up window anchors to naming (Plan 2 Task 1), replacing design section 5 layer 2."

```bash
git add -A && git commit -m "fix(gate): anchor follow-up window to naming, single extension, no re-arm on replies"
```

---

### Task 2: Label CLI

**Files:**
- Create: `familiar_brain/cli.py`
- Modify: `pyproject.toml` (`familiar-brain = "familiar_brain.cli:main"`; move the server entry to `familiar-brain serve`)
- Test: `tests/test_cli_label_report.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli_label_report.py`:
```python
import io
import json
from pathlib import Path
from familiar_brain.cli import label_session, gate_report


def _log(tmp: Path) -> Path:
    p = tmp / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"ts": 1.0, "text": "door creaks", "decision": "IGNORE", "spoke": False},
        {"ts": 2.0, "text": "Understudy, smell?", "decision": "ADDRESSED", "spoke": True},
        {"ts": 3.0, "text": "moving on", "decision": "ADDRESSED", "spoke": True},
        {"ts": 4.0, "text": "little one, roll", "decision": "IGNORE", "spoke": False},
    ]) + "\n")
    return p


def test_label_session_writes_expected_field(tmp_path: Path):
    p = _log(tmp_path)
    answers = io.StringIO("i\na\ni\na\n")   # i=IGNORE a=ADDRESSED m=AMBIENT, enter keeps current
    label_session(p, stdin=answers, stdout=io.StringIO())
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert [r["expected"] for r in rows] == ["IGNORE", "ADDRESSED", "IGNORE", "ADDRESSED"]


def test_gate_report_counts_and_latency(tmp_path: Path):
    p = _log(tmp_path)
    label_session(p, stdin=io.StringIO("i\na\ni\na\n"), stdout=io.StringIO())
    brain_log = tmp_path / "brain.log"
    brain_log.write_text("\n".join([
        json.dumps({"gate_ms": 1, "total_ms": 900, "decision": "ADDRESSED", "spoke": True}),
        json.dumps({"gate_ms": 1, "total_ms": 2100, "decision": "ADDRESSED", "spoke": True}),
        json.dumps({"gate_ms": 1, "total_ms": 3, "decision": "IGNORE", "spoke": False}),
        "not json",
    ]))
    rep = gate_report(p, brain_log)
    assert rep["turns"] == 4 and rep["accuracy"] == 0.5
    assert rep["uninvited_speech"] == 1 and rep["missed_addressed"] == 1
    assert rep["latency_median_ms"] == 1500 and rep["latency_p95_ms"] >= 2000
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli_label_report.py -v`
Expected: FAIL, `familiar_brain.cli` not found.

- [ ] **Step 3: Implement `familiar_brain/cli.py`**

```python
"""familiar-brain CLI: serve, label, gate-report, index."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import TextIO

KEYS = {"a": "ADDRESSED", "m": "AMBIENT", "i": "IGNORE"}


def label_session(path: Path, *, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows:
        stdout.write(f"\n[{r.get('decision')}{' spoke' if r.get('spoke') else ''}] {r['text']}\n")
        stdout.write("expected? a=ADDRESSED m=AMBIENT i=IGNORE, enter=keep: ")
        stdout.flush()
        ans = stdin.readline().strip().lower()
        r["expected"] = KEYS.get(ans, r.get("expected", r.get("decision")))
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def gate_report(session_path: Path, brain_log: Path) -> dict:
    rows = [json.loads(l) for l in Path(session_path).read_text().splitlines() if l.strip()]
    labelled = [r for r in rows if "expected" in r]
    correct = sum(1 for r in labelled if r["expected"] == r["decision"])
    uninvited = sum(1 for r in labelled if r.get("spoke") and r["expected"] != "ADDRESSED")
    missed = sum(1 for r in labelled if r["expected"] == "ADDRESSED" and not r.get("spoke"))
    lat = []
    for l in Path(brain_log).read_text().splitlines():
        try:
            rec = json.loads(l)
        except json.JSONDecodeError:
            continue
        if rec.get("decision") == "ADDRESSED" and rec.get("spoke"):
            lat.append(float(rec["total_ms"]))
    lat.sort()
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))] if lat else None
    return {
        "turns": len(labelled),
        "accuracy": round(correct / len(labelled), 3) if labelled else None,
        "uninvited_speech": uninvited,
        "missed_addressed": missed,
        "latency_median_ms": statistics.median(lat) if lat else None,
        "latency_p95_ms": p95,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="familiar-brain")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("config", nargs="?", default="config/brain.toml")
    l = sub.add_parser("label"); l.add_argument("session")
    g = sub.add_parser("gate-report"); g.add_argument("session"); g.add_argument("brain_log")
    ix = sub.add_parser("index"); ix.add_argument("corpus", choices=["rules", "notes"])
    ix.add_argument("path", nargs="?"); ix.add_argument("--edition", choices=["2014", "2024"])
    ix.add_argument("config", nargs="?", default="config/brain.toml")
    a = p.parse_args(argv)
    if a.cmd == "serve":
        from familiar_brain.app import main as serve_main
        sys.argv = ["familiar-brain", a.config]; serve_main()
    elif a.cmd == "label":
        label_session(Path(a.session))
    elif a.cmd == "gate-report":
        print(json.dumps(gate_report(Path(a.session), Path(a.brain_log)), indent=2))
    elif a.cmd == "index":
        from familiar_brain.indexing import index_command   # Task 8
        index_command(a)
```

- [ ] **Step 4: Update `pyproject.toml` scripts**

```toml
[project.scripts]
familiar-brain = "familiar_brain.cli:main"
```
and change `scripts/start_all.sh` to `uv run familiar-brain serve config/brain.toml`.

- [ ] **Step 5: Run to verify it passes**

Run: `uv sync --extra dev && uv run pytest tests/test_cli_label_report.py -v && uv run familiar-brain --help`
Expected: 2 PASS; help lists serve, label, gate-report, index.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(cli): label and gate-report commands; serve moved under cli"
```

---

### Task 3: Live loop script

**Files:**
- Create: `docs/live-loop.md`

- [ ] **Step 1: Write the document**

```markdown
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
| 4 | Understudy, what do you make of the smell down here? | ADDRESSED |
| 5 | Anything else? | ADDRESSED |
| 6 | Right, Gustave, you are up. | IGNORE |
| 7 | Does anyone remember how grapple works? | IGNORE |
| 8 | Little one, roll me a d20 with advantage. | ADDRESSED |
| 9 | Nice. | IGNORE |
| 10 | Understudy, add Gustave to initiative at fourteen. | ADDRESSED |
| 11 | And Maelle at nine. | ADDRESSED |
| 12 | Who is up first? | ADDRESSED |
| 13 | Okay, the door opens. | IGNORE |
| 14 | (30 seconds of unrelated talking, radio, or a podcast) | IGNORE throughout |

## Tune
- If lines 1, 2, 6, 7, 13 produce no log entry at all: VAD is missing speech, lower `--thresh` by 0.1.
- If the 30 s block produces many one-word segments: raise `--thresh` by 0.1.
- If line 5 or 11 is IGNORE: the follow-up window is too short for your cadence; raise `follow_up_window_s` to 10.
- If line 9 or 13 is ADDRESSED: the window is too long, or the classifier is generous; lower to 6 and check the classifier answer in brain.log.

## Record
```
curl -X POST http://127.0.0.1:8090/api/session/end
uv run familiar-brain label data/sessions/<id>.jsonl
uv run familiar-brain gate-report data/sessions/<id>.jsonl data/brain.log
```
Paste the report and the final `--thresh` and `follow_up_window_s` into docs/versions.md under "Live loop 1". Targets: accuracy 1.0, uninvited_speech 0, latency_median_ms under 1500, latency_p95_ms under 2500.
```

- [ ] **Step 2: Commit**

```bash
git add docs/live-loop.md && git commit -m "docs: scripted live loop and tuning procedure"
```

---

### Task 4: Run the live loop (human in the loop)

- [ ] **Step 1:** Follow `docs/live-loop.md` end to end. This task is done by TJ with Claude Code assisting on log reading.
- [ ] **Step 2:** Paste the gate-report into `docs/versions.md`. If targets are missed, apply the tuning table, re-run once, record both runs.
- [ ] **Step 3:** Keep the labelled `data/sessions/<id>.jsonl` and copy it to `tests/fixtures/live_session_1.jsonl` (it contains only your own voice, so it can live in the repo).
- [ ] **Step 4: Commit**

```bash
git add docs/versions.md tests/fixtures/live_session_1.jsonl && git commit -m "docs: live loop 1 results and real transcript fixture"
```

Part B starts only after this commit exists.

---

## Part B: Knowledge layer

### Task 5: Embedder protocol, CPU embedder, fake embedder

**Files:**
- Create: `familiar_brain/embeddings.py`, `tests/fakes/embedder.py`
- Modify: `pyproject.toml` dependencies: add `sentence-transformers>=3.0`, `sqlite-vec>=0.1.6`, `pymupdf>=1.24`, `numpy>=1.26`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write the fake and the failing test**

`tests/fakes/embedder.py`:
```python
"""Deterministic, dependency-free embedder for tests: hashed bag of words, 64 dims, L2-normalised."""
from __future__ import annotations

import hashlib
import math
import re


class FakeEmbedder:
    dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out
```

`tests/test_embeddings.py`:
```python
import pytest
from tests.fakes.embedder import FakeEmbedder
from familiar_brain.embeddings import Embedder, cosine


def test_fake_embedder_satisfies_protocol_and_similarity_orders():
    e: Embedder = FakeEmbedder()
    a, b, c = e.embed(["grapple the target", "grapple a creature", "cast fireball"])
    assert cosine(a, b) > cosine(a, c)


@pytest.mark.slow
def test_bge_embedder_loads_on_cpu_and_has_384_dims():
    pytest.importorskip("sentence_transformers")
    from familiar_brain.embeddings import BgeEmbedder
    e = BgeEmbedder()
    v = e.embed(["hello"])[0]
    assert len(v) == 384 and e.dim == 384
```

Add to `pyproject.toml` under `[tool.pytest.ini_options]`: `markers = ["slow: needs model download"]` and `addopts = "-m 'not slow'"`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv sync --extra dev && uv run pytest tests/test_embeddings.py -v`
Expected: FAIL, `familiar_brain.embeddings` not found.

- [ ] **Step 3: Implement `familiar_brain/embeddings.py`**

```python
from __future__ import annotations

import math
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class BgeEmbedder:
    """CPU-only sentence-transformers embedder. Model pinned; loaded lazily on first call."""
    MODEL = "BAAI/bge-small-en-v1.5"
    dim = 384

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL, device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._load().encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vecs]
```

- [ ] **Step 4: Run to verify; run the slow test once by hand**

Run: `uv run pytest tests/test_embeddings.py -v` then `uv run pytest tests/test_embeddings.py -m slow -v`
Expected: fast test PASS; slow test PASS after the one-time model download (about 130 MB). Note the model revision in `docs/versions.md`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(knowledge): Embedder protocol, CPU bge-small embedder, fake embedder for tests"
```

---

### Task 6: Store over SQLite plus sqlite-vec

**Files:**
- Create: `familiar_brain/retrieval.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_retrieval.py`:
```python
from pathlib import Path
from familiar_brain.retrieval import Store, Chunk
from tests.fakes.embedder import FakeEmbedder


def make(tmp: Path) -> Store:
    return Store(tmp / "k.db", embedder=FakeEmbedder())


def test_add_and_search_returns_best_match_with_meta(tmp_path):
    s = make(tmp_path)
    s.add_chunks([
        Chunk(corpus="rules", source="srd51.pdf", title="Grappling", text="When you want to grab a creature you can use the Attack action to make a special melee attack, a grapple.", meta={"edition": "2014"}),
        Chunk(corpus="rules", source="srd51.pdf", title="Fireball", text="A bright streak flashes from your pointing finger to a point you choose.", meta={"edition": "2014"}),
        Chunk(corpus="notes", source="pc.md", title="Backstory", text="My character grew up on the docks of Nephria.", meta={}),
    ])
    hits = s.search("how does grapple work", corpus="rules", k=1)
    assert hits[0].title == "Grappling" and hits[0].meta["edition"] == "2014" and hits[0].score > 0


def test_search_is_scoped_to_corpus(tmp_path):
    s = make(tmp_path)
    s.add_chunks([Chunk(corpus="notes", source="a.md", title="A", text="grapple grapple grapple", meta={})])
    assert s.search("grapple", corpus="rules", k=3) == []


def test_delete_source_removes_its_chunks_only(tmp_path):
    s = make(tmp_path)
    s.add_chunks([
        Chunk(corpus="notes", source="a.md", title="A", text="alpha", meta={}),
        Chunk(corpus="notes", source="b.md", title="B", text="beta", meta={}),
    ])
    s.delete_source("a.md")
    assert [h.title for h in s.search("alpha beta", corpus="notes", k=5)] == ["B"]


def test_weight_scales_score(tmp_path):
    s = make(tmp_path)
    s.add_chunks([
        Chunk(corpus="sessions", source="s1.summary.md", title="Summary", text="the barkeep owes gold", meta={}, weight=1.5),
        Chunk(corpus="sessions", source="s1.jsonl", title="Raw", text="the barkeep owes gold", meta={}, weight=1.0),
    ])
    hits = s.search("barkeep gold", corpus="sessions", k=2)
    assert hits[0].title == "Summary" and hits[0].score > hits[1].score
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `familiar_brain/retrieval.py`**

```python
from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path

import sqlite_vec

from familiar_brain.embeddings import Embedder


@dataclass
class Chunk:
    corpus: str
    source: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class Hit:
    corpus: str
    source: str
    title: str
    text: str
    meta: dict
    score: float


def _pack(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


class Store:
    def __init__(self, path: Path | str, *, embedder: Embedder):
        self.embedder = embedder
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.executescript(f"""
        CREATE TABLE IF NOT EXISTS chunks(
          id INTEGER PRIMARY KEY, corpus TEXT NOT NULL, source TEXT NOT NULL, title TEXT,
          text TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{{}}', weight REAL NOT NULL DEFAULT 1.0,
          created_at REAL DEFAULT (unixepoch('subsec')));
        CREATE INDEX IF NOT EXISTS chunks_corpus ON chunks(corpus);
        CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
        CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{embedder.dim}]);
        """)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vecs = self.embedder.embed([f"{c.title}\n{c.text}" for c in chunks])
        with self.db:
            for c, v in zip(chunks, vecs):
                cur = self.db.execute(
                    "INSERT INTO chunks(corpus, source, title, text, meta, weight) VALUES (?,?,?,?,?,?)",
                    (c.corpus, c.source, c.title, c.text, json.dumps(c.meta), c.weight))
                self.db.execute("INSERT INTO vec(chunk_id, embedding) VALUES (?,?)", (cur.lastrowid, _pack(v)))
        return len(chunks)

    def search(self, query: str, *, corpus: str, k: int = 3) -> list[Hit]:
        q = self.embedder.embed([query])[0]
        rows = self.db.execute("""
            SELECT c.corpus, c.source, c.title, c.text, c.meta, c.weight, v.distance
            FROM vec v JOIN chunks c ON c.id = v.chunk_id
            WHERE v.embedding MATCH ? AND k = ? AND c.corpus = ?
            ORDER BY v.distance""", (_pack(q), max(k * 8, 32), corpus)).fetchall()
        hits = [Hit(r[0], r[1], r[2], r[3], json.loads(r[4]), score=(1.0 - r[6] / 2.0) * r[5]) for r in rows]
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    def delete_source(self, source: str) -> int:
        with self.db:
            ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE source = ?", (source,))]
            if ids:
                qs = ",".join("?" * len(ids))
                self.db.execute(f"DELETE FROM vec WHERE chunk_id IN ({qs})", ids)
                self.db.execute(f"DELETE FROM chunks WHERE id IN ({qs})", ids)
        return len(ids)

    def sources(self, corpus: str) -> dict[str, float]:
        """source -> max created_at, used by the notes indexer to skip unchanged files."""
        return {r[0]: r[1] for r in self.db.execute(
            "SELECT source, MAX(created_at) FROM chunks WHERE corpus = ? GROUP BY source", (corpus,))}
```

Note on the `WHERE ... k = ?` form: sqlite-vec's KNN query syntax has changed across versions (`k = ?` versus `LIMIT ?`). Check `uv run python -c "import sqlite_vec; print(sqlite_vec.__version__)"` against the sqlite-vec docs and use whichever the installed version accepts; the test tells you immediately. Corpus filtering happens in SQL after the KNN candidate set, so `k * 8` candidates are pulled to keep recall when corpora are mixed.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(knowledge): sqlite-vec store with corpus scoping, weights, and source deletion"
```

---

### Task 7: Chunkers

**Files:**
- Create: `familiar_brain/chunking.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_chunking.py`:
```python
from pathlib import Path
import fitz  # pymupdf
from familiar_brain.chunking import markdown_chunks, pdf_chunks_by_bookmarks, split_soft


def test_markdown_chunks_by_heading_carry_title():
    md = "# Backstory\nDocks of Nephria.\n\n## Contacts\nThe barkeep.\nOwes gold.\n"
    out = markdown_chunks(md, source="pc.md", corpus="notes")
    assert [c.title for c in out] == ["Backstory", "Contacts"]
    assert out[1].text.startswith("The barkeep") and out[1].source == "pc.md"


def test_split_soft_respects_cap_and_sentence_boundary():
    text = "Sentence one is here. " * 100
    parts = split_soft(text, cap=300)
    assert all(len(p) <= 330 for p in parts) and all(p.rstrip().endswith(".") for p in parts)


def test_pdf_chunks_use_bookmarks_and_edition(tmp_path: Path):
    doc = fitz.open()
    for i, (title, body) in enumerate([("Grappling", "You can grapple. " * 20), ("Fireball", "A bright streak. " * 20)]):
        page = doc.new_page(); page.insert_text((72, 72), body)
    doc.set_toc([[1, "Grappling", 1], [1, "Fireball", 2]])
    p = tmp_path / "srd.pdf"; doc.save(p)
    out = pdf_chunks_by_bookmarks(p, corpus="rules", edition="2014")
    assert [c.title for c in out] == ["Grappling", "Fireball"]
    assert all(c.meta["edition"] == "2014" for c in out) and "grapple" in out[0].text.lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `familiar_brain/chunking.py`**

```python
from __future__ import annotations

import re
from pathlib import Path

import fitz

from familiar_brain.retrieval import Chunk

SOFT_CAP = 1200


def split_soft(text: str, cap: int = SOFT_CAP) -> list[str]:
    """Split on sentence ends so pieces stay under cap (plus one sentence of slack)."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    parts, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > cap:
            parts.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def markdown_chunks(md: str, *, source: str, corpus: str) -> list[Chunk]:
    out: list[Chunk] = []
    title, buf = Path(source).stem, []
    def flush():
        body = "\n".join(buf).strip()
        if body:
            for piece in split_soft(body):
                out.append(Chunk(corpus=corpus, source=source, title=title, text=piece))
    for line in md.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            flush(); buf = []; title = m.group(1).strip()
        else:
            buf.append(line)
    flush()
    return out


def pdf_chunks_by_bookmarks(path: Path | str, *, corpus: str, edition: str) -> list[Chunk]:
    doc = fitz.open(str(path))
    toc = [(t, max(pg, 1)) for _lvl, t, pg in doc.get_toc()]
    if not toc:
        toc = [(f"Page {i + 1}", i + 1) for i in range(len(doc))]
    out: list[Chunk] = []
    for i, (title, start) in enumerate(toc):
        end = toc[i + 1][1] if i + 1 < len(toc) else len(doc)
        text = " ".join(doc[p].get_text() for p in range(start - 1, max(end, start)))
        text = re.sub(r"\s+", " ", text).strip()
        # trim the leading text that belongs to the previous section when two bookmarks share a page
        idx = text.find(title)
        if 0 < idx < 400:
            text = text[idx:]
        for piece in split_soft(text):
            out.append(Chunk(corpus=corpus, source=Path(path).name, title=title, text=piece,
                             meta={"edition": edition, "page": start}))
    return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(knowledge): bookmark-based PDF chunker and markdown chunker"
```

---

### Task 8: Indexer CLI for rules and notes

**Files:**
- Create: `familiar_brain/indexing.py`
- Modify: `familiar_brain/config.py`, `config/brain.toml` (add `[knowledge]`), `README.md`
- Test: `tests/test_indexing.py`

- [ ] **Step 1: Add config**

`config/brain.toml`:
```toml
[knowledge]
db_path = "data/familiar.db"
notes_dir = "data/notes"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
```
In `config.py` add:
```python
@dataclass(frozen=True)
class Knowledge:
    db_path: str
    notes_dir: str
    chunks_per_corpus: int
    max_context_chars: int
    retention_days: int
```
add `knowledge: Knowledge` to `Config` and `knowledge=Knowledge(**raw["knowledge"])` in `load_config`. Update `tests/test_config.py` and the integration fixture's TOML to include the section.

- [ ] **Step 2: Write the failing test**

`tests/test_indexing.py`:
```python
from pathlib import Path
from familiar_brain.indexing import index_notes
from familiar_brain.retrieval import Store
from tests.fakes.embedder import FakeEmbedder


def test_index_notes_skips_unchanged_and_reindexes_changed(tmp_path: Path):
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "pc.md").write_text("# PC\nDocks of Nephria.\n")
    (notes / "npcs.md").write_text("# Barkeep\nOwes gold.\n")
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    assert index_notes(store, notes) == {"indexed": 2, "skipped": 0}
    assert index_notes(store, notes) == {"indexed": 0, "skipped": 2}
    import os, time
    (notes / "pc.md").write_text("# PC\nNow lives uptown.\n")
    os.utime(notes / "pc.md", (time.time() + 5, time.time() + 5))
    assert index_notes(store, notes) == {"indexed": 1, "skipped": 1}
    assert store.search("uptown", corpus="notes", k=1)[0].text.startswith("Now lives")
    assert not any("Docks" in h.text for h in store.search("docks", corpus="notes", k=5))  # old chunk gone
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_indexing.py -v`
Expected: FAIL, module not found.

- [ ] **Step 4: Implement `familiar_brain/indexing.py`**

```python
from __future__ import annotations

from pathlib import Path

from familiar_brain.chunking import markdown_chunks, pdf_chunks_by_bookmarks
from familiar_brain.retrieval import Store


def index_rules(store: Store, pdf: Path, *, edition: str) -> int:
    store.delete_source(Path(pdf).name)
    return store.add_chunks(pdf_chunks_by_bookmarks(pdf, corpus="rules", edition=edition))


def index_notes(store: Store, notes_dir: Path) -> dict:
    known = store.sources("notes")
    indexed = skipped = 0
    for md in sorted(Path(notes_dir).glob("**/*.md")):
        src = str(md.relative_to(notes_dir))
        if src in known and md.stat().st_mtime <= known[src]:
            skipped += 1; continue
        store.delete_source(src)
        store.add_chunks(markdown_chunks(md.read_text(encoding="utf-8"), source=src, corpus="notes"))
        indexed += 1
    return {"indexed": indexed, "skipped": skipped}


def open_store(cfg) -> Store:
    from familiar_brain.embeddings import BgeEmbedder
    Path(cfg.knowledge.db_path).parent.mkdir(parents=True, exist_ok=True)
    return Store(cfg.knowledge.db_path, embedder=BgeEmbedder())


def index_command(args) -> None:
    from familiar_brain.config import load_config
    cfg = load_config(args.config)
    store = open_store(cfg)
    if args.corpus == "rules":
        if not args.path or not args.edition:
            raise SystemExit("index rules <pdf> --edition 2014|2024")
        n = index_rules(store, Path(args.path), edition=args.edition)
        print(f"indexed {n} rule chunks from {args.path} (edition {args.edition})")
    else:
        print(index_notes(store, Path(args.path or cfg.knowledge.notes_dir)))
```

- [ ] **Step 5: Run the test, then index the real SRDs**

Run: `uv run pytest tests/test_indexing.py -v`
Expected: PASS.

Download both PDFs from https://www.dndbeyond.com/srd into `data/rules/` (the 5.1 and the latest 5.2.x). Then:
```bash
uv run familiar-brain index rules data/rules/SRD_CC_v5.1.pdf --edition 2014
uv run familiar-brain index rules data/rules/SRD_CC_v5.2.1.pdf --edition 2024
uv run python -c "
from familiar_brain.config import load_config; from familiar_brain.indexing import open_store
s=open_store(load_config('config/brain.toml'))
for h in s.search('grapple', corpus='rules', k=4): print(h.meta['edition'], h.title, h.text[:80])"
```
Expected: chunk counts in the low thousands per PDF, and grapple hits from both editions. Adjust filenames to what you downloaded.

- [ ] **Step 6: Add attribution to `README.md`**

```markdown
## Rules attribution
This work includes material from the System Reference Document 5.1 and 5.2.1 ("SRD 5.1", "SRD 5.2.1") by Wizards of the Coast LLC, available at https://www.dndbeyond.com/srd. The SRDs are licensed under the Creative Commons Attribution 4.0 International License, available at https://creativecommons.org/licenses/by/4.0/legalcode.
```

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(knowledge): rules and notes indexer CLI; SRD attribution"
```

---

### Task 9: `lookup_rule` and `recall` tools

**Files:**
- Create: `familiar_brain/tools/lookup_rule.py`, `familiar_brain/tools/recall.py`
- Test: `tests/test_lookup_recall.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_lookup_recall.py`:
```python
from pathlib import Path
from familiar_brain.retrieval import Store, Chunk
from familiar_brain.tools import lookup_rule, recall
from tests.fakes.embedder import FakeEmbedder


def store(tmp: Path) -> Store:
    s = Store(tmp / "k.db", embedder=FakeEmbedder())
    s.add_chunks([
        Chunk("rules", "srd51.pdf", "Grappling", "Grapple 2014 text about grabbing a creature.", {"edition": "2014"}),
        Chunk("rules", "srd52.pdf", "Grappling", "Grapple 2024 text, unarmed strike grapple option.", {"edition": "2024"}),
        Chunk("notes", "npcs.md", "Barkeep", "The barkeep owes the party forty gold.", {}),
        Chunk("sessions", "s1.summary.md", "Session 1", "The party met the barkeep at the Gilded Eel.", {}, 1.5),
    ])
    return s


def test_lookup_rule_returns_one_per_edition_and_names_it(tmp_path):
    lookup_rule.bind(store(tmp_path))
    out = lookup_rule.run({"query": "grapple a creature"})
    assert "2014 rules" in out and "2024 rules" in out and out.count("Grappling") == 2


def test_lookup_rule_empty_store_is_graceful(tmp_path):
    lookup_rule.bind(Store(tmp_path / "e.db", embedder=FakeEmbedder()))
    assert "nothing" in lookup_rule.run({"query": "x"}).lower()


def test_recall_scopes(tmp_path):
    s = store(tmp_path); recall.bind(s)
    assert "forty gold" in recall.run({"query": "barkeep gold", "scope": "notes"})
    assert "Gilded Eel" in recall.run({"query": "barkeep", "scope": "sessions"})
    both = recall.run({"query": "barkeep"})
    assert "forty gold" in both and "Gilded Eel" in both
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_lookup_recall.py -v`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement `familiar_brain/tools/lookup_rule.py`**

```python
from __future__ import annotations

from familiar_brain.retrieval import Store

_STORE: Store | None = None


def bind(store: Store) -> None:
    global _STORE
    _STORE = store


SPEC = {
    "type": "function",
    "name": "lookup_rule",
    "description": "Look up a Dungeons & Dragons rule. Returns the best passage from the 2014 rules and from the 2024 rules; say which edition you are citing and defer to the Dungeon Master.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}


def run(args: dict) -> str:
    if _STORE is None:
        return "The rulebooks are not open."
    hits = _STORE.search(str(args.get("query", "")), corpus="rules", k=6)
    best = {}
    for h in hits:
        ed = h.meta.get("edition", "?")
        if ed not in best:
            best[ed] = h
    if not best:
        return "I found nothing on that in the rulebooks."
    return "\n".join(f"{ed} rules, {h.title}: {h.text[:600]}" for ed, h in sorted(best.items()))
```

- [ ] **Step 4: Implement `familiar_brain/tools/recall.py`**

```python
from __future__ import annotations

from familiar_brain.retrieval import Store

_STORE: Store | None = None


def bind(store: Store) -> None:
    global _STORE
    _STORE = store


SPEC = {
    "type": "function",
    "name": "recall",
    "description": "Search the campaign notes and past session summaries for something the party has seen, met, decided, or been told.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "scope": {"type": "string", "enum": ["notes", "sessions", "all"]}},
        "required": ["query"],
    },
}


def run(args: dict) -> str:
    if _STORE is None:
        return "My memory is not open."
    scope = args.get("scope", "all")
    corpora = ["notes", "sessions"] if scope == "all" else [scope]
    lines = []
    for c in corpora:
        for h in _STORE.search(str(args.get("query", "")), corpus=c, k=3):
            lines.append(f"[{c}] {h.title}: {h.text[:400]}")
    return "\n".join(lines) if lines else "I recall nothing about that."
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_lookup_recall.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(tools): lookup_rule (both editions) and recall"
```

---

### Task 10: "What you know" injection on ADDRESSED turns

**Files:**
- Create: `familiar_brain/knowledge.py`
- Modify: `familiar_brain/app.py` (`create_app`, the ADDRESSED branch), `familiar_brain/persona.py` (accept an optional context block)
- Test: `tests/test_knowledge.py`, `tests/test_app_knowledge.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_knowledge.py`:
```python
from familiar_brain.knowledge import build_context
from familiar_brain.retrieval import Store, Chunk
from tests.fakes.embedder import FakeEmbedder


def test_build_context_caps_per_corpus_and_total(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    s.add_chunks([Chunk("rules", "r.pdf", f"Rule {i}", "grapple " * 200, {"edition": "2014"}) for i in range(6)])
    s.add_chunks([Chunk("notes", "n.md", "Note", "grapple in the notes", {})])
    ctx = build_context(s, "how does grapple work", chunks_per_corpus=3, max_chars=1500)
    assert ctx.startswith("What you know:") and ctx.count("Rule ") <= 3 and len(ctx) <= 1600
    assert "Note" in ctx or len(ctx) >= 1400  # notes included unless the cap was hit first


def test_build_context_empty_store_returns_empty_string(tmp_path):
    s = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    assert build_context(s, "anything", chunks_per_corpus=3, max_chars=1000) == ""
```

`tests/test_app_knowledge.py` (reuse the Plan 1 integration fixture; add `[knowledge]` to its TOML with `db_path` under `tmp_path`, and monkeypatch `familiar_brain.indexing.open_store` to return a `Store` with `FakeEmbedder`):
```python
import json
from pathlib import Path
import pytest
from familiar_brain import indexing
from familiar_brain.app import create_app
from familiar_brain.retrieval import Store, Chunk
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm_server import FakeLLM


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    fake = FakeLLM()
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    store.add_chunks([Chunk("rules", "srd51.pdf", "Grappling", "Grapple rules text 2014.", {"edition": "2014"})])
    monkeypatch.setattr(indexing, "open_store", lambda cfg: store)
    (tmp_path / "persona.yaml").write_text("name: Understudy\naliases: []\ntts_speaker: A\nmanner: dry\n")
    (tmp_path / "notes").mkdir()
    (tmp_path / "brain.toml").write_text(f"""
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
persona = "{tmp_path / 'persona.yaml'}"
sessions_dir = "{tmp_path / 'sessions'}"
[knowledge]
db_path = "{tmp_path / 'k.db'}"
notes_dir = "{tmp_path / 'notes'}"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
""")
    app = create_app(tmp_path / "brain.toml"); app.config["TESTING"] = True
    yield app.test_client(), fake
    fake.close()


def test_addressed_turn_carries_what_you_know(env):
    client, fake = env
    client.post("/api/session/start")
    client.post("/v1/responses", json={"input": [{"role": "user", "content": "Understudy, how does grapple work?"}], "stream": True, "tools": []})
    instr = fake.requests[-1]["body"]["instructions"]
    assert "What you know:" in instr and "Grappling" in instr and instr.index("What you know:") < instr.index("You are Understudy")


def test_ignored_turn_does_no_retrieval(env):
    client, fake = env
    client.post("/api/session/start")
    client.post("/v1/responses", json={"input": [{"role": "user", "content": "the door creaks"}], "stream": True})
    assert not any(r["path"].endswith("/responses") for r in fake.requests)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_knowledge.py tests/test_app_knowledge.py -v`
Expected: FAIL, `familiar_brain.knowledge` not found.

- [ ] **Step 3: Implement `familiar_brain/knowledge.py`**

```python
from __future__ import annotations

from familiar_brain.retrieval import Store

CORPORA = ("rules", "notes", "sessions")
CHUNK_CHARS = 700  # per-chunk cap so one long rule never crowds out the notes


def build_context(store: Store, query: str, *, chunks_per_corpus: int, max_chars: int) -> str:
    lines: list[str] = []
    used = 0
    for corpus in CORPORA:
        for h in store.search(query, corpus=corpus, k=chunks_per_corpus):
            tag = f" ({h.meta['edition']} rules)" if corpus == "rules" and "edition" in h.meta else ""
            line = f"- [{corpus}] {h.title}{tag}: {h.text[:CHUNK_CHARS]}"
            if used + len(line) > max_chars:
                return _wrap(lines)
            lines.append(line); used += len(line)
    return _wrap(lines)


def _wrap(lines: list[str]) -> str:
    return "What you know:\n" + "\n".join(lines) if lines else ""
```

- [ ] **Step 4: Modify `persona.render_system_prompt`** to accept `context: str = ""` and, when non-empty, return `context + "\n\n" + <existing prompt>`. Plan 1 tests keep passing because the default is empty.

- [ ] **Step 5: Modify `create_app`**

Read the current function. Add, after the persona is loaded:
```python
    from familiar_brain import indexing
    from familiar_brain.knowledge import build_context
    from familiar_brain.tools import lookup_rule as lookup_tool, recall as recall_tool
    store = indexing.open_store(cfg)
    indexing.index_notes(store, Path(cfg.knowledge.notes_dir)) if Path(cfg.knowledge.notes_dir).exists() else None
    lookup_tool.bind(store); recall_tool.bind(store)
```
register `lookup_tool` and `recall_tool` alongside the existing three, and in the ADDRESSED branch replace the fixed `instructions=system_prompt` with:
```python
        try:
            context = build_context(store, text, chunks_per_corpus=cfg.knowledge.chunks_per_corpus,
                                    max_chars=cfg.knowledge.max_context_chars)
        except Exception as e:  # retrieval failure: proceed without context
            app.logger.warning("retrieval failed: %s", e); context = ""
        timer.mark("retrieval")
        outgoing = with_instructions_and_tools(req, instructions=render_system_prompt(persona, context=context),
                                               extra_tools=tools.specs())
```
Also add `POST /api/reindex` returning `jsonify(indexing.index_notes(store, Path(cfg.knowledge.notes_dir)))`, and re-run notes indexing inside `api_session_start` before `log.start()`.

- [ ] **Step 6: Run everything**

Run: `uv run pytest -q`
Expected: all green, including Plan 1's `test_persona_instructions_and_tools_reach_upstream` now also seeing `lookup_rule` and `recall` in tools.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(brain): inject retrieved context on ADDRESSED turns; notes reindex at session start and /api/reindex"
```

---

### Task 11: Summariser, retention, purge

**Files:**
- Create: `familiar_brain/summariser.py`
- Modify: `familiar_brain/app.py` (`session/end` starts the job; `DELETE /api/session/<id>`; `GET /api/sessions`; retention at start), `familiar_brain/llm.py` (add `complete(messages, max_tokens)` on the generate slot, non-stream)
- Test: `tests/test_summariser.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_summariser.py`:
```python
import json
import time
from pathlib import Path
from familiar_brain.retrieval import Store
from familiar_brain.summariser import SUMMARY_PROMPT, summarise_session, apply_retention, purge_session
from tests.fakes.embedder import FakeEmbedder


class FakeLLM:
    def __init__(self): self.calls = []
    def complete(self, messages, max_tokens=1200):
        self.calls.append(messages)
        return "## Scenes\n- The cistern\n## NPCs met\n- The barkeep\n## Decisions\n- Went down\n## Loot\n- none\n## Open threads\n- Who owes gold?\n## Memorable lines\n- The smell of old coin."


def _session(tmp: Path) -> Path:
    p = tmp / "sessions" / "20260911-1.jsonl"; p.parent.mkdir(parents=True)
    p.write_text("\n".join(json.dumps(r) for r in [
        {"ts": 1, "text": "The cistern is dark.", "decision": "IGNORE", "spoke": False},
        {"ts": 2, "text": "Understudy, smell?", "decision": "ADDRESSED", "spoke": True},
    ]) + "\n")
    return p


def test_prompt_forbids_player_names_and_ooc():
    assert "player" in SUMMARY_PROMPT.lower() and "out of character" in SUMMARY_PROMPT.lower()


def test_summarise_writes_file_and_indexes_both(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder()); llm = FakeLLM()
    p = _session(tmp_path)
    out = summarise_session(p, store=store, llm=llm)
    assert out == p.with_suffix(".summary.md") and "barkeep" in out.read_text()
    hits = store.search("barkeep", corpus="sessions", k=2)
    assert hits and hits[0].source.endswith(".summary.md")
    assert any(h.source.endswith(".jsonl") for h in store.search("cistern dark", corpus="sessions", k=3))


def test_retention_deletes_old_raw_keeps_summary(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    p = _session(tmp_path); summarise_session(p, store=store, llm=FakeLLM())
    old = time.time() - 10 * 86400
    import os; os.utime(p, (old, old))
    removed = apply_retention(p.parent, store=store, retention_days=7)
    assert removed == [p.name] and not p.exists() and p.with_suffix(".summary.md").exists()
    assert all(not h.source.endswith(".jsonl") for h in store.search("cistern", corpus="sessions", k=5))


def test_purge_removes_everything_for_session(tmp_path):
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    p = _session(tmp_path); summarise_session(p, store=store, llm=FakeLLM())
    assert purge_session(p.parent, "20260911-1", store=store) is True
    assert not p.exists() and not p.with_suffix(".summary.md").exists()
    assert store.search("barkeep", corpus="sessions", k=3) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_summariser.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Add `LLMClient.complete`** in `llm.py`

```python
    def complete(self, messages: list[dict], max_tokens: int = 1200) -> str:
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.2, "stream": False,
                "id_slot": self.generate_slot}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
```
`LLMClient.__init__` gains `generate_slot: int = 0`. Pass `cfg.upstream.generate_slot` from `create_app`. If Plan 1 disabled Gemma's reasoning via a request field or chat template kwarg, apply the same field here.

- [ ] **Step 4: Implement `familiar_brain/summariser.py`**

```python
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from familiar_brain.chunking import markdown_chunks, split_soft
from familiar_brain.retrieval import Chunk, Store

log = logging.getLogger("familiar.summariser")

SUMMARY_PROMPT = (
    "You are the memory of a familiar at a Dungeons & Dragons table. Below is a transcript of what was said "
    "during one session, one line per utterance. Write a summary of the FICTION only, in Markdown with exactly "
    "these headings: ## Scenes, ## NPCs met, ## Decisions, ## Loot, ## Open threads, ## Memorable lines.\n"
    "Rules: never name or describe the players or anything said out of character; do not record dice results, "
    "table logistics, or jokes about the players; keep each bullet under 25 words; if a heading has nothing, "
    "write '- none'."
)


def summarise_session(path: Path, *, store: Store, llm) -> Path:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    transcript = "\n".join(r["text"] for r in rows if r.get("text"))
    md = llm.complete([{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": transcript}])
    out = Path(path).with_suffix(".summary.md")
    out.write_text(md, encoding="utf-8")
    sid = Path(path).stem
    store.delete_source(out.name); store.delete_source(Path(path).name)
    summary_chunks = [Chunk("sessions", out.name, f"Session {sid}: {c.title}", c.text, {"session": sid}, weight=1.5)
                      for c in markdown_chunks(md, source=out.name, corpus="sessions")]
    raw_chunks = [Chunk("sessions", Path(path).name, f"Session {sid} transcript", piece, {"session": sid}, weight=1.0)
                  for piece in split_soft(transcript)]
    store.add_chunks(summary_chunks + raw_chunks)
    return out


def summarise_in_background(path: Path, *, store: Store, llm) -> threading.Thread:
    def job():
        try:
            summarise_session(path, store=store, llm=llm)
            log.info("summarised %s", path.name)
        except Exception as e:
            log.error("summariser failed for %s: %s", path.name, e)
    t = threading.Thread(target=job, daemon=True, name="summariser"); t.start()
    return t


def apply_retention(sessions_dir: Path, *, store: Store, retention_days: int) -> list[str]:
    cutoff = time.time() - retention_days * 86400
    removed = []
    for raw in Path(sessions_dir).glob("*.jsonl"):
        if raw.stat().st_mtime < cutoff:
            store.delete_source(raw.name); raw.unlink(); removed.append(raw.name)
    return removed


def purge_session(sessions_dir: Path, session_id: str, *, store: Store) -> bool:
    raw = Path(sessions_dir) / f"{session_id}.jsonl"; summ = raw.with_suffix(".summary.md")
    found = False
    for f in (raw, summ):
        if f.exists():
            store.delete_source(f.name); f.unlink(); found = True
    return found
```

- [ ] **Step 5: Wire into `app.py`**

In `create_app`, after the store exists: `apply_retention(Path(cfg.paths.sessions_dir), store=store, retention_days=cfg.knowledge.retention_days)`. Change `api_session_end` to:
```python
        path = log.end()
        summarise_in_background(path, store=store, llm=llm)
        return jsonify(path=str(path), summarising=True)
```
(keep Plan 1's no-active-session guard). Add:
```python
    @app.delete("/api/session/<sid>")
    def api_session_purge(sid: str):
        return jsonify(purged=purge_session(Path(cfg.paths.sessions_dir), sid, store=store))

    @app.get("/api/sessions")
    def api_sessions():
        d = Path(cfg.paths.sessions_dir)
        ids = sorted({p.stem.replace(".summary", "") for p in d.glob("*")})
        return jsonify(sessions=[{"id": i, "summary": (d / f"{i}.summary.md").exists(), "raw": (d / f"{i}.jsonl").exists()} for i in ids])
```

- [ ] **Step 6: Run everything, then summarise the real live-loop transcript**

Run: `uv run pytest -q`
Expected: all green.

Then, with llama-server up:
```bash
uv run python -c "
from pathlib import Path
from familiar_brain.config import load_config; from familiar_brain.indexing import open_store; from familiar_brain.llm import LLMClient
from familiar_brain.summariser import summarise_session
cfg=load_config('config/brain.toml'); s=open_store(cfg)
llm=LLMClient(base_url=cfg.upstream.base_url, classify_slot=cfg.upstream.classify_slot, classify_timeout_s=1, generate_slot=cfg.upstream.generate_slot)
print(summarise_session(Path('tests/fixtures/live_session_1.jsonl'), store=s, llm=llm).read_text())"
```
Read the output against the fiction-only rule. If it mentions you as a player, dice totals, or logistics, tighten `SUMMARY_PROMPT` and re-run. Record the final output in `docs/versions.md` under "Summariser check".

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(knowledge): fiction-only session summariser, retention, purge, sessions listing"
```

---

### Task 12: Close out

- [ ] **Step 1:** Update `docs/deviations.md` with every place the current code differed from this plan's snippets.
- [ ] **Step 2:** Dash audit on authored files: `grep -rnP "[\x{2014}\x{2013}]" docs familiar_brain tests README.md || echo clean` (use a Python check if grep's PCRE lacks wide code points).
- [ ] **Step 3:** `uv run pytest -q` green; `uv run pytest -m slow -q` green once.
- [ ] **Step 4:** Live check: start the stack, start a session, ask "Understudy, how does grapple work in the new rules?" and confirm the spoken answer names 2024 and the brain log shows `retrieval_ms`. End the session and confirm `.summary.md` appears within five minutes.
- [ ] **Step 5:** Commit and open the branch for review.

```bash
git add -A && git commit -m "chore: plan 2 close-out, deviations and verification recorded"
```

---

## Self-review against the spec

Goals 1 to 3 are Tasks 2, 3, 4 (label, report, scripted run, targets in the doc). Goal 4 is Tasks 7, 8, 9 (both SRDs, edition tag, `lookup_rule` names the edition). Goal 5 is Tasks 8, 9, 10 (`recall`, notes indexing at session start and `/api/reindex`). Goal 6 is Task 11 (background summariser, indexed with weight 1.5, retention, purge). Decisions section: gate anchoring (Task 1), CPU embeddings (Task 5), SRDs by bookmark (Tasks 7, 8), no watcher (Task 8), fiction-only prompt (Task 11). P1 endpoints `/api/reindex` (Task 10) and `/api/sessions` (Task 11). Acceptance: integration test for "What you know" (Task 10), live results in `versions.md` (Tasks 4, 11, 12).

Type consistency: `Store(path, embedder=...)`, `Chunk(corpus, source, title, text, meta, weight)` positional order is used identically in Tasks 6, 7, 9, 10, 11 tests. `Hit.meta`, `Hit.score`, `Hit.source` used consistently. `LLMClient.complete(messages, max_tokens)` matches the summariser's `FakeLLM`. `indexing.open_store(cfg)` is the single seam the app and CLI share and the integration test monkeypatches. `render_system_prompt(persona, context="")` keeps the Plan 1 call sites valid.
