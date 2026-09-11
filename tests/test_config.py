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
