"""TDD tests for:
1. Native reachy_familiar robot mode (eliminating the false 'robot disabled (FAMILIAR_NO_ROBOT=1)' bug).
2. Switchable Campaign / Team Profiles with isolated roster, notes, campaign memory,
   speaker profiles, privacy names, and non-leaking system prompts.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from familiar_brain.app import create_app
from familiar_brain.embeddings import make_embedder
from familiar_brain.retrieval import Store


def _write_min_config(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    notes_dir = data_dir / "notes"
    sessions_dir = data_dir / "sessions"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    (notes_dir / "nephria_lore.md").write_text(
        "# Nephria\nThe city of Nephria runs on Art Deco lamps and ink.\n",
        encoding="utf-8",
    )
    (cfg_dir / "persona.yaml").write_text(
        "name: Rocky\naliases: [rocky]\ntts_speaker: bm_fable\nmanner: Dry and alert.\ngestures:\n  attentive: play_emotion:attentive\n",
        encoding="utf-8",
    )
    (cfg_dir / "lexicon.tsv").write_text("Nephria\tnefrea\n", encoding="utf-8")
    (cfg_dir / "players.txt").write_text("James\nElias\n", encoding="utf-8")
    cfg = cfg_dir / "brain.toml"
    cfg.write_text(
        f"""
[server]
host = "127.0.0.1"
port = 8090

[upstream]
base_url = "http://127.0.0.1:9"
generate_slot = 0
classify_slot = 1
classify_timeout_s = 1.0

[gate]
follow_up_window_s = 8.0
max_follow_ups = 2
llm_classifier = false

[paths]
persona = "{(cfg_dir / 'persona.yaml').as_posix()}"
lexicon = "{(cfg_dir / 'lexicon.tsv').as_posix()}"
sessions_dir = "{sessions_dir.as_posix()}"
players = "{(cfg_dir / 'players.txt').as_posix()}"

[knowledge]
db_path = "{(data_dir / 'knowledge.sqlite').as_posix()}"
notes_dir = "{notes_dir.as_posix()}"
embedder = "deterministic"
chunks_per_corpus = 2
max_context_chars = 1800
retention_days = 30

[speakers]
enabled = true
store_path = "{(data_dir / 'speakers.json').as_posix()}"
""",
        encoding="utf-8",
    )
    return cfg


@pytest.fixture()
def native_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = _write_min_config(tmp_path)
    monkeypatch.setenv("FAMILIAR_ROBOT_MODE", "native")
    monkeypatch.delenv("FAMILIAR_NO_ROBOT", raising=False)
    monkeypatch.delenv("ROBOT_HOST", raising=False)
    monkeypatch.setenv("FAMILIAR_EMBEDDER", "deterministic")
    with patch("familiar_brain.indexing.open_store") as mock_open:
        store = Store(tmp_path / "data" / "knowledge.sqlite", embedder=make_embedder("deterministic"))
        mock_open.return_value = store
        app = create_app(cfg)
        yield app
        app.extensions["close"]()


def test_native_robot_mode_never_reports_disabled_before_or_after_heartbeat(native_app, monkeypatch: pytest.MonkeyPatch):
    import requests

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"ok": True, "detail": "Restarting Reachy audio link..."}

    # Reconnect now forwards to the robot's :8043 app; keep the test off the network.
    monkeypatch.setattr(requests.Session, "post", lambda self, url, *a, **k: _Resp())
    client = native_app.test_client()

    # Before heartbeat arrives (while voice engine is still warming up): must NOT say FAMILIAR_NO_ROBOT=1.
    st0 = client.get("/api/robot/status").get_json()
    assert st0["disabled"] is False
    assert "FAMILIAR_NO_ROBOT" not in st0["detail"]
    assert st0["mode"] == "native"

    # After reachy_familiar sends a heartbeat, /api/robot/status keeps the live link instead of clobbering it.
    hb = client.post(
        "/api/robot/heartbeat",
        json={"backend_connected": True, "host": "reachy-mini.local", "app": "reachy_familiar", "mic_level": 0.25},
    )
    assert hb.status_code == 200
    st1 = client.get("/api/robot/status").get_json()
    assert st1["backend_connected"] is True
    assert st1["disabled"] is False
    assert "reachy_familiar" in st1["detail"]

    # Clicking Reconnect in native mode must NOT return 503 robot disabled.
    rc = client.post("/api/robot/reconnect", json={})
    assert rc.status_code == 200
    assert rc.get_json()["disabled"] is False


def test_heartbeat_overrides_stale_no_robot_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Even if FAMILIAR_NO_ROBOT=1 was set, once reachy_familiar heartbeats, /api/robot/status stays connected."""
    cfg = _write_min_config(tmp_path)
    monkeypatch.setenv("FAMILIAR_NO_ROBOT", "1")
    monkeypatch.delenv("FAMILIAR_ROBOT_MODE", raising=False)
    with patch("familiar_brain.indexing.open_store") as mock_open:
        store = Store(tmp_path / "data" / "knowledge.sqlite", embedder=make_embedder("deterministic"))
        mock_open.return_value = store
        app = create_app(cfg)
        try:
            client = app.test_client()
            client.post(
                "/api/robot/heartbeat",
                json={"backend_connected": True, "host": "reachy-mini.local", "app": "reachy_familiar"},
            )
            st = client.get("/api/robot/status").get_json()
            assert st["backend_connected"] is True
            assert st["disabled"] is False
            assert "FAMILIAR_NO_ROBOT" not in st["detail"]
        finally:
            app.extensions["close"]()


def test_dynamic_party_roster_and_privacy_names_live_updates(native_app):
    client = native_app.test_client()

    # Default campaign exposes roster and privacy names from players.txt
    r0 = client.get("/api/campaigns").get_json()
    assert r0["active"]["id"] == "nephria"
    assert "DM" in r0["active"]["roster"]
    assert "Brokk" in r0["active"]["roster"]
    assert "James" in r0["active"]["privacy_names"]

    # Add a new player/character mid-session and a new real first name to strip from summaries
    r_add = client.post("/api/roster", json={"action": "add", "role": "Vex"}).get_json()
    assert r_add["ok"] is True
    assert "Vex" in r_add["roster"]

    r_rm = client.post("/api/roster", json={"action": "remove", "role": "Till"}).get_json()
    assert r_rm["ok"] is True
    assert "Till" not in r_rm["roster"]

    p_add = client.post("/api/privacy-names", json={"action": "add", "name": "Taylor"}).get_json()
    assert p_add["ok"] is True
    assert "Taylor" in p_add["privacy_names"]

    p_rm = client.post("/api/privacy-names", json={"action": "remove", "name": "Elias"}).get_json()
    assert p_rm["ok"] is True
    assert "Elias" not in p_rm["privacy_names"]

    # /api/state includes active campaign roster so UI chips stay in sync over SSE
    st = client.get("/api/state").get_json()
    assert st["campaign"]["id"] == "nephria"
    assert "Vex" in st["campaign"]["roster"]


def test_switch_campaign_profile_isolates_memory_notes_roster_and_prompt(native_app):
    client = native_app.test_client()

    # Save Nephria-specific campaign memory
    client.put("/api/memory", json={"campaign_memory": "Brokk met Lady Vespera in Nephria."})
    assert "Nephria" in client.get("/api/memory").get_json()["campaign_memory"]

    # Create a brand new team/campaign profile ("Curse of Strahd")
    created = client.post(
        "/api/campaigns",
        json={
            "name": "Curse of Strahd",
            "setting": "the gothic mists of Barovia",
            "bound_to": "Seraphina",
            "roster": ["DM", "Seraphina", "Garrick", "Nyx"],
            "secrets": "Never mention the Sunsword location unprompted.",
        },
    )
    assert created.status_code == 200
    cid = created.get_json()["profile"]["id"]
    assert cid == "curse-of-strahd"

    # Switch to the new team profile
    sw = client.post("/api/campaigns/switch", json={"id": cid})
    assert sw.status_code == 200
    active = sw.get_json()["active"]
    assert active["id"] == cid
    assert active["roster"] == ["DM", "Seraphina", "Garrick", "Nyx"]

    # Memory is isolated (Nephria memory does NOT bleed into Barovia)
    mem_barovia = client.get("/api/memory").get_json()["campaign_memory"]
    assert "Nephria" not in mem_barovia
    assert "Vespera" not in mem_barovia

    # Write Barovia-only memory
    client.put("/api/memory", json={"campaign_memory": "The party entered Vallaki at dusk."})
    assert "Vallaki" in client.get("/api/memory").get_json()["campaign_memory"]

    # System prompt adapts to Barovia & Seraphina and does NOT leak Nephria secrets (Vacoran Amulet / Inkdrinker)
    from familiar_brain.persona import render_system_prompt

    prompt = render_system_prompt(
        native_app.extensions["persona"],
        campaign_profile=native_app.extensions["campaign_manager"].active(),
    )
    assert "Barovia" in prompt
    assert "Seraphina" in prompt
    assert "Garrick" in prompt
    assert "Vacoran Amulet" not in prompt
    assert "Inkdrinker" not in prompt

    # Switch back to Nephria -> original Nephria memory and secrets are completely intact!
    sw_back = client.post("/api/campaigns/switch", json={"id": "nephria"})
    assert sw_back.status_code == 200
    assert "Lady Vespera" in client.get("/api/memory").get_json()["campaign_memory"]
    nephria_prompt = render_system_prompt(
        native_app.extensions["persona"],
        campaign_profile=native_app.extensions["campaign_manager"].active(),
    )
    assert "Vacoran Amulet" in nephria_prompt
    assert "Nephria" in nephria_prompt


def test_campaign_switch_keeps_session_active(native_app):
    client = native_app.test_client()
    started = client.post("/api/session/start")
    assert started.status_code == 200
    assert started.get_json()["session"]

    created = client.post("/api/campaigns", json={"name": "Lost Mine", "roster": ["DM", "Nyx"]})
    assert created.status_code == 200
    cid = created.get_json()["profile"]["id"]
    assert client.post("/api/campaigns/switch", json={"id": cid}).status_code == 200

    st = client.get("/api/state").get_json()
    # Switching mid-game used to end the session and start none, so nothing was logged after.
    assert st["session"], "a session active before the switch must stay active after it"
    sessions_dir = native_app.extensions["campaign_manager"].active().sessions_dir
    assert (sessions_dir / f"{st['session']}.jsonl").exists()


def test_robot_reconnect_route_reaches_native_app(native_app, monkeypatch: pytest.MonkeyPatch):
    import requests

    calls: list[str] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"ok": True, "detail": "Restarting Reachy audio link..."}

    real_post = requests.Session.post

    def fake_post(self, url, *args, **kwargs):
        if ":8043/" in str(url):
            calls.append(str(url))
            return _Resp()
        return real_post(self, url, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "post", fake_post)
    client = native_app.test_client()
    resp = client.post("/api/robot/reconnect", json={})
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True
    # The native app restarts its audio link only when its :8043 route is called.
    assert calls == ["http://reachy-mini.local:8043/_familiar/reconnect-audio"]


def test_native_reconnect_prefers_launcher_robot_host(native_app, monkeypatch: pytest.MonkeyPatch):
    # WSL2 often cannot resolve .local names; launch.ps1 resolves the robot and exports ROBOT_HOST.
    import requests

    calls: list[str] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"ok": True}

    monkeypatch.setenv("ROBOT_HOST", "10.0.0.77")
    monkeypatch.setattr(requests.Session, "post", lambda self, url, *a, **k: calls.append(str(url)) or _Resp())
    resp = native_app.test_client().post("/api/robot/reconnect", json={})
    assert resp.status_code == 200
    assert calls == ["http://10.0.0.77:8043/_familiar/reconnect-audio"]
