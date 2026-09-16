from pathlib import Path
from familiar_brain.config import load_config

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
[knowledge]
db_path = "data/familiar.db"
notes_dir = "data/notes"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
""")
    c = load_config(p)
    assert c.server.port == 8090
    assert c.upstream.base_url.endswith("/v1")
    assert c.upstream.classify_timeout_s == 0.6
    assert c.gate.follow_up_window_s == 8
    assert c.paths.sessions_dir == "data/sessions"
    assert c.paths.lexicon == "config/lexicon.tsv"  # defaulted, absent from this TOML
    assert c.paths.players == "config/players.txt"  # defaulted too
    assert c.knowledge.db_path == "data/familiar.db"
    assert c.knowledge.chunks_per_corpus == 3
    assert c.knowledge.retention_days == 7


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
