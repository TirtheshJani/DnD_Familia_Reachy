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
