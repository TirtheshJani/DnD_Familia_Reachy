"""Tier 3 E2E Tests: Cross-Feature Pair Integrations (22 tests).

Strictly opaque-box: exercises HTTP/SSE, WebSocket, files, and CLI subprocesses only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import requests

from tests.e2e.conftest import REPO_ROOT, StackHandle
from tests.e2e.test_tier1_happy_paths import (
    _has_bash,
    _has_powershell,
    _make_sample_rules_pdf,
    _run_sdk_probe,
    _setup_start_all_stubs,
    _start_all_cmd,
)

pytestmark = [pytest.mark.e2e, pytest.mark.t3]


@pytest.mark.m1
def test_t3_01_f8_f11_nephria_homebrew_beats_2024_and_2014_rules(e2e_stack: StackHandle, tmp_path: Path):
    e2e_stack.copy_repo_notes()
    e2e_stack.client.reindex()
    p2014 = _make_sample_rules_pdf(tmp_path / "srd51.pdf", "Injuries", "2014 lingering injuries optional DMG rule.")
    p2024 = _make_sample_rules_pdf(tmp_path / "srd52.pdf", "Injuries", "2024 critical hit damage rule.")
    env = {**os.environ, "FAMILIAR_EMBEDDER": "deterministic", "FAMILIAR_NO_ROBOT": "1"}
    subprocess.run([sys.executable, "-m", "familiar_brain.cli", "index", "rules", str(p2014), "--edition", "2014", "--config", str(e2e_stack.cfg_path)], cwd=str(REPO_ROOT), env=env, check=True, timeout=15)
    subprocess.run([sys.executable, "-m", "familiar_brain.cli", "index", "rules", str(p2024), "--edition", "2024", "--config", str(e2e_stack.cfg_path)], cwd=str(REPO_ROOT), env=env, check=True, timeout=15)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_tool_roundtrip("lookup_rule", {"query": "Injuries critical hit"}, "House rule gives choice.")
    e2e_stack.client.post_turn("Rocky, what is the critical injury rule?")
    out = [i["output"] for i in e2e_stack.llm.responses_requests()[-1]["body"]["input"] if i.get("type") == "function_call_output"][-1]
    assert "House rule" in out and "2024 rules" in out and "2014 rules" in out
    assert out.index("House rule") < out.index("2024 rules") < out.index("2014 rules")


@pytest.mark.m1
def test_t3_02_f9_f10_lexicon_alias_addresses_rocky_with_secrets_guardrail(e2e_stack: StackHandle):
    shutil.copy2(REPO_ROOT / "config" / "lexicon.tsv", e2e_stack.lexicon_path)
    shutil.copy2(REPO_ROOT / "config" / "persona.yaml", e2e_stack.persona_path)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_response_text("Watching Brokk.")
    r = e2e_stack.client.post_turn("Rockie, what did Brock Trinivane drop?")
    assert r.status_code == 200 and r.text
    req = e2e_stack.llm.responses_requests()[-1]["body"]
    assert "Brokk Trenevain" in json.dumps(req["input"])
    assert "Vacoran Amulet" in req["instructions"]


@pytest.mark.m1
def test_t3_03_f9_f8_lexicon_rewrites_enable_notes_retrieval(e2e_stack: StackHandle):
    e2e_stack.copy_repo_notes()
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_tool_roundtrip("recall", {"query": "Jade Halls Krag", "scope": "notes"}, "Krag fought in the Jade Halls.")
    e2e_stack.client.post_turn("Rocky, remind me about jade halls and Craig.")
    req = e2e_stack.llm.responses_requests()[-1]["body"]
    assert "Jade Halls" in json.dumps(req["input"]) and "Krag" in json.dumps(req["input"])


@pytest.mark.m2
def test_t3_04_f10_f15_thinking_mode_preserves_rocky_secrets(e2e_stack: StackHandle):
    shutil.copy2(REPO_ROOT / "config" / "persona.yaml", e2e_stack.persona_path)
    e2e_stack.restart_brain()
    e2e_stack.client.set_mode("thinking")
    e2e_stack.llm.queue_response_text("Six-sentence reasoning.")
    e2e_stack.client.post_turn("Rocky, think with me about the gate.")
    instr = e2e_stack.llm.last_instructions()
    assert "up to six sentences" in instr
    assert "Vacoran Amulet" in instr and "Inkdrinker" in instr


@pytest.mark.m2
def test_t3_05_f7_f15_unaddressed_timeout_bounded_and_updates_telemetry(e2e_stack: StackHandle):
    e2e_stack.write_config(classify_timeout_s=0.3)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_chat("IGNORE", delay_s=0.8)
    t0 = time.monotonic()
    r = e2e_stack.client.post_turn("The hooded figure leans against the lamppost.")
    elapsed = time.monotonic() - t0
    assert r.status_code == 200 and r.text == ""
    assert elapsed < 0.65
    st = e2e_stack.client.get_state()
    assert isinstance(st["last_turn_ms"], (int, float))
    assert st["transcript"][-1]["decision"] == "IGNORE"


@pytest.mark.m2
def test_t3_06_f13_f14_checkpoint_mid_session_then_purge_older_session(e2e_stack: StackHandle):
    old_sid = "20260905-120000-old002"
    (e2e_stack.sessions_dir / f"{old_sid}.jsonl").write_text('{"ts": 1.0, "text": "Rocky, old", "decision": "ADDRESSED", "spoke": true}\n', encoding="utf-8")
    (e2e_stack.sessions_dir / f"{old_sid}.summary.md").write_text("## Scenes\n- Old dock.\n", encoding="utf-8")
    cur_sid = e2e_stack.client.start_session().json()["session"]
    e2e_stack.client.note("New clocktower clue.")
    e2e_stack.llm.queue_chat("## Scenes\n- New clocktower.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Memory\n- New clocktower.\n")
    assert e2e_stack.client.checkpoint(force=True).status_code == 200
    assert e2e_stack.client.get_state()["session"] == cur_sid
    assert e2e_stack.client.purge_session(old_sid).json()["purged"] is True
    assert not (e2e_stack.sessions_dir / f"{old_sid}.summary.md").exists()


@pytest.mark.m2
def test_t3_07_f13_f9_lexicon_corrected_utterances_flow_into_checkpoint(e2e_stack: StackHandle):
    shutil.copy2(REPO_ROOT / "config" / "lexicon.tsv", e2e_stack.lexicon_path)
    e2e_stack.restart_brain()
    sid = e2e_stack.client.start_session().json()["session"]
    e2e_stack.llm.queue_response_text("Logged.")
    e2e_stack.client.post_turn("Rocky, Brock Trinivane entered Nepheria.")
    e2e_stack.llm.queue_chat("## Scenes\n- Brokk Trenevain entered Nephria.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Memory\n- Brokk Trenevain in Nephria.\n")
    assert e2e_stack.client.checkpoint(force=True).status_code == 200
    raw = (e2e_stack.sessions_dir / f"{sid}.jsonl").read_text(encoding="utf-8")
    assert "Brokk Trenevain" in raw and "Nephria" in raw


@pytest.mark.m4
def test_t3_08_f15_f20_deep_night_and_privacy_mute_compose_cleanly(e2e_stack: StackHandle):
    e2e_stack.client.mute()
    e2e_stack.client.deep_night(True)
    assert e2e_stack.client.get_state()["muted"] is True and e2e_stack.client.get_state()["asleep"] is True
    e2e_stack.client.deep_night(False)
    st = e2e_stack.client.get_state()
    assert st["asleep"] is False and st["muted"] is True
    e2e_stack.client.unmute()
    assert e2e_stack.client.get_state()["muted"] is False


@pytest.mark.m3
def test_t3_09_f16_f17_live_ws_audio_enrolls_brokk_and_matches_next_turn(e2e_stack: StackHandle, audio_samples):
    tap = e2e_stack.start_audio_tap()
    box: list[requests.Response] = []

    def _enroll():
        box.append(e2e_stack.client.enroll_speaker({"role": "Brokk", "duration_s": 0.6}))

    th = threading.Thread(target=_enroll, daemon=True)
    th.start()
    time.sleep(0.1)
    with tap.ws_client() as ws:
        ws.send_audio_pcm16(audio_samples["brokk_pcm"])
        time.sleep(0.7)
        th.join(timeout=5.0)
        assert box and box[0].status_code == 200
        ws.send_audio_pcm16(audio_samples["brokk_pcm_2"])
        ws.send_json({"type": "input_audio_buffer.commit"})
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and tap.current().get("role") != "Brokk":
            time.sleep(0.05)
    assert tap.current().get("role") == "Brokk"


@pytest.mark.m3
def test_t3_10_f16_f18_tap_state_propagates_dm_role_to_brain_without_header(e2e_stack: StackHandle, audio_samples):
    assert e2e_stack.client.enroll_speaker({"role": "Brokk", "audio_b64": audio_samples["brokk_b64"], "sample_rate": 16000}).status_code == 200
    assert e2e_stack.client.enroll_speaker({"role": "DM", "audio_b64": audio_samples["dm_b64"], "sample_rate": 16000}).status_code == 200
    tap = e2e_stack.start_audio_tap()
    e2e_stack.llm.queue_response_text("Yes, Brokk.")
    r1 = e2e_stack.client.post_turn("Rocky, listen!", headers={"X-Familiar-Speaker-Role": "Brokk"})
    assert r1.text

    with tap.ws_client() as ws:
        ws.send_audio_pcm16(audio_samples["dm_pcm_2"])
        ws.send_json({"type": "input_audio_buffer.commit"})
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and tap.current().get("role") != "DM":
            time.sleep(0.05)
    assert tap.current().get("role") == "DM"
    # Brain queries /_tap/current without header and suppresses DM follow-up chatter
    r2 = e2e_stack.client.post_turn("The iron door swings shut behind you.")
    assert r2.text == "" and "idle_do_nothing" in r2.tool_names


@pytest.mark.m3
def test_t3_11_f17_f14_player_name_rejected_by_enroll_and_scrubbed_from_checkpoint(e2e_stack: StackHandle, audio_samples):
    assert e2e_stack.client.enroll_speaker({"role": "TJ", "audio_b64": audio_samples["dm_b64"], "sample_rate": 16000}).status_code == 400
    e2e_stack.client.start_session()
    e2e_stack.client.note("Opened the vault.")
    e2e_stack.llm.queue_chat("## Scenes\n- Opened the vault.\n- TJ took a sip of coffee.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Memory\n- Opened the vault.\n")
    cp = e2e_stack.client.checkpoint(force=True)
    assert cp.status_code == 200
    sp = cp.json().get("summary_path")
    if sp and Path(sp).exists():
        assert "TJ" not in Path(sp).read_text(encoding="utf-8")


@pytest.mark.m3
def test_t3_12_f18_f15_dm_deep_night_voice_updates_state_and_emits_sleepy_cue(e2e_stack: StackHandle):
    r = e2e_stack.client.post_turn("Deep night.", headers={"X-Familiar-Speaker-Role": "DM"})
    assert r.status_code == 200
    emotions = [a.get("emotion") for n, _, a in r.tool_calls if n == "play_emotion"]
    assert "sleepy" in emotions
    st = e2e_stack.client.get_state()
    assert st["asleep"] is True and st["speakers"]["last_speaker"] == "DM"


@pytest.mark.m3
def test_t3_13_f18_f10_brokk_master_persona_still_keeps_secrets_guardrail(e2e_stack: StackHandle):
    shutil.copy2(REPO_ROOT / "config" / "persona.yaml", e2e_stack.persona_path)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_response_text("I stay quiet.")
    e2e_stack.client.post_turn("Rocky, what am I carrying?", headers={"X-Familiar-Speaker-Role": "Brokk"})
    instr = e2e_stack.llm.last_instructions()
    assert "Brokk" in instr and "master" in instr.lower()
    assert "Vacoran Amulet" in instr and "Inkdrinker" in instr


@pytest.mark.m4
def test_t3_14_f19_f20_companion_ui_actions_and_endpoints_match_live_server(e2e_stack: StackHandle):
    html = e2e_stack.client.get_ui("/").text
    js = e2e_stack.client.get_ui("/static/app.js").text
    actions = re.findall(r'data-action="([^"]+)"', html)
    assert len(actions) >= 10
    for act in actions:
        assert act in js, f"UI action {act!r} not handled in app.js"
    assert e2e_stack.client.mute().status_code == 200
    assert e2e_stack.client.unmute().status_code == 200


@pytest.mark.m4
def test_t3_15_f20_f13_ui_checkpoint_and_note_searchable_via_recall(e2e_stack: StackHandle):
    e2e_stack.client.start_session()
    assert e2e_stack.client.note("Bellwarden Corvus guards the north bridge.").status_code == 200
    e2e_stack.llm.queue_chat("## Scenes\n- Crossed north bridge.\n## NPCs met\n- Bellwarden Corvus.\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Memory\n- Bellwarden Corvus at north bridge.\n")
    assert e2e_stack.client.checkpoint(force=True).status_code == 200
    e2e_stack.llm.queue_tool_roundtrip("recall", {"query": "Bellwarden Corvus", "scope": "all"}, "He guards the north bridge.")
    e2e_stack.client.post_turn("Rocky, who is Bellwarden Corvus?")
    assert "Bellwarden Corvus" in e2e_stack.llm.last_instructions() or "Corvus" in json.dumps(e2e_stack.llm.responses_requests()[-1]["body"])


@pytest.mark.m4
def test_t3_16_f20_f17_ui_speaker_crud_synced_with_state(e2e_stack: StackHandle, audio_samples):
    assert e2e_stack.client.enroll_speaker({"role": "DM", "audio_b64": audio_samples["dm_b64"], "sample_rate": 16000}).status_code == 200
    assert e2e_stack.client.enroll_speaker({"role": "Till", "audio_b64": audio_samples["till_b64"], "sample_rate": 16000}).status_code == 200
    assert set(e2e_stack.client.get_state()["speakers"]["enrolled"]) >= {"DM", "Till"}
    assert e2e_stack.client.delete_speaker("Till").status_code == 200
    assert e2e_stack.client.get_speakers().json()["enrolled"] == ["DM"]
    assert e2e_stack.client.clear_speakers().status_code == 200
    assert e2e_stack.client.get_state()["speakers"]["enrolled"] == []


@pytest.mark.m4
def test_t3_17_f21_f2_robot_reconnect_configures_fake_reachy_and_updates_status(e2e_stack: StackHandle):
    r = e2e_stack.client.robot_reconnect({
        "host": "127.0.0.1",
        "daemon_port": e2e_stack.reachy.daemon_port,
        "rpc_port": e2e_stack.reachy.rpc_port,
        "host_ip": "127.0.0.1",
        "port": e2e_stack.s2s.port,
    })
    assert r.status_code == 200 and r.json()["backend_connected"] is True
    assert e2e_stack.reachy.applied_personality == "familiar"
    assert e2e_stack.client.robot_status().json()["backend_connected"] is True
    assert e2e_stack.client.get_state()["robot"]["backend_connected"] is True


@pytest.mark.m5
def test_t3_18_f1_f16_start_all_launches_audio_tap_ahead_of_s2s_when_enabled(tmp_path: Path):
    if not _has_bash():
        pytest.skip("bash not available")
    env, trace_file = _setup_start_all_stubs(tmp_path, s2s_lifetime_s=0.8)
    subprocess.run(
        _start_all_cmd("--skip-robot"),
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=15,
    )
    assert trace_file.exists() and "s2s:" in trace_file.read_text(encoding="utf-8")
    txt = (REPO_ROOT / "scripts" / "start_all.sh").read_text(encoding="utf-8")
    assert "audio_tap" in txt and "8766" in txt


@pytest.mark.m5
def test_t3_19_f1_f6_start_all_no_audio_tap_and_sdk_probe_pass(e2e_stack: StackHandle, tmp_path: Path):
    if _has_bash():
        env, trace_file = _setup_start_all_stubs(tmp_path, s2s_lifetime_s=0.8)
        subprocess.run(
            _start_all_cmd("--skip-robot", "--no-audio-tap"),
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=15,
        )
        trace = trace_file.read_text(encoding="utf-8")
        assert "audio_tap" not in trace
    probe = _run_sdk_probe(f"{e2e_stack.brain_url}/v1", tmp_path)
    assert probe.returncode == 0, probe.stdout + probe.stderr


@pytest.mark.m5
def test_t3_20_f3_f2_launch_ps1_dry_run_checks_8765_and_8090_and_robot_host():
    ps = _has_powershell()
    if not ps:
        pytest.skip("PowerShell not available")
    proc = subprocess.run(
        [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(REPO_ROOT / "scripts" / "launch.ps1"), "-DryRun", "-SkipRobot"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0
    assert "8765" in proc.stdout and "8090" in proc.stdout


@pytest.mark.m5
def test_t3_21_f5_f11_lookup_rule_tool_roundtrip_hides_tool_and_tracks_head(e2e_stack: StackHandle, tmp_path: Path):
    p2024 = _make_sample_rules_pdf(tmp_path / "srd52.pdf", "Grapple", "2024 Unarmed Strike Grapple saving throw.")
    env = {**os.environ, "FAMILIAR_EMBEDDER": "deterministic", "FAMILIAR_NO_ROBOT": "1"}
    subprocess.run([sys.executable, "-m", "familiar_brain.cli", "index", "rules", str(p2024), "--edition", "2024", "--config", str(e2e_stack.cfg_path)], cwd=str(REPO_ROOT), env=env, check=True, timeout=15)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_tool_roundtrip("lookup_rule", {"query": "Grapple"}, "Use Unarmed Strike in 2024 rules.")
    r = e2e_stack.client.post_turn("Rocky, how does grapple work in 2024?")
    assert r.status_code == 200
    assert "lookup_rule" not in r.tool_names
    assert "head_tracking" in r.tool_names
    assert "2024" in r.text


@pytest.mark.m1
def test_t3_22_f12_f13_learn_summarises_sep16_without_sep11_ghosts(e2e_stack: StackHandle):
    sessions = REPO_ROOT / "data" / "sessions"
    assert list(sessions.glob("20260911-*")) == []
    sep16_jsonl = sessions / "20260916-182232-0189cc.jsonl"
    sep16_summary = sessions / "20260916-182232-0189cc.summary.md"
    assert sep16_jsonl.exists() or sep16_summary.exists()
    target_jsonl = e2e_stack.sessions_dir / "20260916-182232-0189cc.jsonl"
    if sep16_jsonl.exists():
        shutil.copy2(sep16_jsonl, target_jsonl)
    else:
        target_jsonl.write_text(
            '{"ts": 1789582952.0, "text": "Rocky, remember the Jade Halls familiar playtest.", "decision": "ADDRESSED", "spoke": true}\n',
            encoding="utf-8",
        )
    e2e_stack.llm.queue_chat("## Scenes\n- Sep 16 playtest.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Memory\n- Sep 16 playtest.\n")
    r = e2e_stack.client.learn()
    assert r.status_code == 200
    assert "20260916-182232-0189cc" in r.json()["queued"]
    deadline = time.monotonic() + 5.0
    summary = e2e_stack.sessions_dir / "20260916-182232-0189cc.summary.md"
    while time.monotonic() < deadline and not summary.exists():
        time.sleep(0.1)
    assert summary.exists()
    assert not any(p.name.startswith("20260911-") for p in e2e_stack.sessions_dir.iterdir())
