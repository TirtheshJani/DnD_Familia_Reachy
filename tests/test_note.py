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
