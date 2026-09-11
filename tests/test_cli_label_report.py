import io
import json
from pathlib import Path

from familiar_brain.cli import gate_report, label_session


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


def test_label_session_enter_keeps_the_gate_decision(tmp_path: Path):
    p = _log(tmp_path)
    label_session(p, stdin=io.StringIO("\n\n\n\n"), stdout=io.StringIO())
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert [r["expected"] for r in rows] == ["IGNORE", "ADDRESSED", "ADDRESSED", "IGNORE"]


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


def test_gate_report_with_no_spoken_turns_reports_none_not_a_crash(tmp_path: Path):
    p = _log(tmp_path)
    label_session(p, stdin=io.StringIO("i\ni\ni\ni\n"), stdout=io.StringIO())
    brain_log = tmp_path / "brain.log"
    brain_log.write_text(json.dumps({"decision": "IGNORE", "spoke": False, "total_ms": 3}) + "\n")
    rep = gate_report(p, brain_log)
    assert rep["latency_median_ms"] is None and rep["latency_p95_ms"] is None


def test_gate_report_says_so_when_the_brain_log_is_missing(tmp_path: Path):
    """docs/live-loop.md has you type the brain log path by hand, after labelling is
    already done. A traceback there loses the report; a silent None would read as
    "latency was fine". Report the counts and say the log was missing."""
    p = _log(tmp_path)
    label_session(p, stdin=io.StringIO("i\na\ni\na\n"), stdout=io.StringIO())
    rep = gate_report(p, tmp_path / "nope.log")
    assert rep["turns"] == 4 and rep["accuracy"] == 0.5 and rep["uninvited_speech"] == 1
    assert rep["brain_log_missing"] is True
    assert rep["latency_median_ms"] is None and rep["latency_p95_ms"] is None


def test_gate_report_omits_the_missing_flag_when_the_log_is_there(tmp_path: Path):
    p = _log(tmp_path)
    label_session(p, stdin=io.StringIO("i\na\ni\na\n"), stdout=io.StringIO())
    brain_log = tmp_path / "brain.log"
    brain_log.write_text(json.dumps({"total_ms": 900, "decision": "ADDRESSED", "spoke": True}) + "\n")
    assert "brain_log_missing" not in gate_report(p, brain_log)
