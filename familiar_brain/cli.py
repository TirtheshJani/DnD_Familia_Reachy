"""familiar-brain CLI: serve, label, gate-report, index."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, TextIO

KEYS = {"a": "ADDRESSED", "m": "AMBIENT", "i": "IGNORE"}


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def label_session(path: Path, *, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    """Walk a session log line by line, recording what the gate SHOULD have decided.

    Writes the answer back to each row as `expected`. Pressing enter keeps whatever
    is already there, so a part-labelled log can be finished in a second pass.
    """
    rows = _rows(path)
    for r in rows:
        stdout.write(f"\n[{r.get('decision')}{' spoke' if r.get('spoke') else ''}] {r['text']}\n")
        stdout.write("expected? a=ADDRESSED m=AMBIENT i=IGNORE, enter=keep: ")
        stdout.flush()
        ans = stdin.readline().strip().lower()
        r["expected"] = KEYS.get(ans, r.get("expected", r.get("decision")))
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                          encoding="utf-8")


def gate_report(session_path: Path, brain_log: Path) -> dict[str, Any]:
    """Score a labelled session against the gate, and read latency from the brain log.

    uninvited_speech is the number that matters: the design's headline criterion is
    that the familiar speaks zero times when it was not addressed.
    """
    labelled = [r for r in _rows(session_path) if "expected" in r]
    correct = sum(1 for r in labelled if r["expected"] == r["decision"])
    uninvited = sum(1 for r in labelled if r.get("spoke") and r["expected"] != "ADDRESSED")
    missed = sum(1 for r in labelled if r["expected"] == "ADDRESSED" and not r.get("spoke"))
    lat: list[float] = []
    log_path = Path(brain_log)
    # The brain log is written by the running service, which lives in WSL, and its path
    # is typed by hand per docs/live-loop.md. Losing the whole report to a typo, after
    # labelling is already done, is worse than reporting the counts without latency.
    missing = not log_path.exists()
    if not missing:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # the brain log carries plain log lines beside the timing records
            if rec.get("decision") == "ADDRESSED" and rec.get("spoke"):
                lat.append(float(rec["total_ms"]))
    lat.sort()
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))] if lat else None
    report = {
        "turns": len(labelled),
        "accuracy": round(correct / len(labelled), 3) if labelled else None,
        "uninvited_speech": uninvited,
        "missed_addressed": missed,
        "latency_median_ms": statistics.median(lat) if lat else None,
        "latency_p95_ms": p95,
    }
    if missing:
        # Explicit, so a None latency cannot be read as "latency was fine".
        report["brain_log_missing"] = True
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="familiar-brain")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the brain HTTP service")
    s.add_argument("config", nargs="?", default="config/brain.toml")
    lab = sub.add_parser("label", help="label a session log with the expected gate decision")
    lab.add_argument("session")
    g = sub.add_parser("gate-report", help="score a labelled session and report latency")
    g.add_argument("session"); g.add_argument("brain_log")
    ix = sub.add_parser("index", help="index rules PDFs or campaign notes into the knowledge store")
    ix.add_argument("corpus", choices=["rules", "notes"])
    ix.add_argument("path", nargs="?")
    ix.add_argument("--edition", choices=["2014", "2024"])
    ix.add_argument("--config", default="config/brain.toml")
    a = p.parse_args(argv)
    if a.cmd == "serve":
        from familiar_brain.app import main as serve_main
        sys.argv = ["familiar-brain", a.config]
        serve_main()
    elif a.cmd == "label":
        label_session(Path(a.session))
    elif a.cmd == "gate-report":
        print(json.dumps(gate_report(Path(a.session), Path(a.brain_log)), indent=2))
    elif a.cmd == "index":
        from familiar_brain.indexing import index_command
        index_command(a)
