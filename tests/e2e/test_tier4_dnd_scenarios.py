"""Tier 4 E2E Tests: Real-World Shadows over Nephria D&D Table Scenarios (13 tests).

Strictly opaque-box: exercises HTTP/SSE, WebSocket, files, and CLI subprocesses only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.e2e.conftest import REPO_ROOT, StackHandle
from tests.e2e.test_tier1_happy_paths import (
    _has_bash,
    _has_powershell,
    _make_sample_rules_pdf,
    _run_sdk_probe,
    _setup_start_all_stubs,
    _start_all_cmd,
)

pytestmark = [pytest.mark.e2e, pytest.mark.t4]


@pytest.mark.m5
def test_t4_01_one_click_preflight_startup_probe_and_fake_robot_connect(e2e_stack: StackHandle, tmp_path: Path):
    ps = _has_powershell()
    if ps:
        ps_proc = subprocess.run(
            [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(REPO_ROOT / "scripts" / "launch.ps1"), "-DryRun", "-SkipRobot"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        assert ps_proc.returncode == 0
    if _has_bash():
        env, trace_file = _setup_start_all_stubs(tmp_path, s2s_lifetime_s=0.8)
        subprocess.run(
            _start_all_cmd("--skip-robot", "--auto-session"),
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=15,
        )
        assert "/api/session/start" in trace_file.read_text(encoding="utf-8")
    probe = _run_sdk_probe(f"{e2e_stack.brain_url}/v1", tmp_path)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    rc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "robot_connect.py"),
            "--robot", "127.0.0.1",
            "--daemon-port", str(e2e_stack.reachy.daemon_port),
            "--rpc-port", str(e2e_stack.reachy.rpc_port),
            "--host-ip", "127.0.0.1",
            "--port", str(e2e_stack.s2s.port),
        ],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
    )
    assert rc.returncode == 0, rc.stderr + rc.stdout
    assert e2e_stack.reachy.backend_connected is True


@pytest.mark.m2
def test_t4_02_full_session_lifecycle_play_break_checkpoint_end_and_purge(e2e_stack: StackHandle):
    e2e_stack.write_config(follow_up_window_s=0.1)
    shutil.copy2(REPO_ROOT / "config" / "lexicon.tsv", e2e_stack.lexicon_path)
    e2e_stack.restart_brain()
    sid = e2e_stack.client.start_session().json()["session"]
    assert isinstance(e2e_stack.client.get_state()["session_started_at"], (int, float))

    e2e_stack.llm.queue_response_text("Watching Old Town.")
    assert e2e_stack.client.post_turn("Rocky, Brock Trinivane is scouting Old Town.").text
    time.sleep(0.25)
    assert e2e_stack.client.post_turn("Pass me a d6.").text == ""
    e2e_stack.llm.chat_replies.clear()
    e2e_stack.client.note("Met the lantern-maker in Old Town.")

    e2e_stack.llm.queue_chat("## Scenes\n- Brokk Trenevain scouted Old Town.\n## NPCs met\n- Lantern-maker.\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Memory\n- Scouted Old Town.\n")
    cp = e2e_stack.client.checkpoint(force=True)
    assert cp.status_code == 200 and e2e_stack.client.get_state()["session"] == sid

    e2e_stack.llm.queue_response_text("Good find!")
    assert e2e_stack.client.post_turn("Rocky, we found the jade sigil after the break.").text

    e2e_stack.llm.queue_chat("## Scenes\n- Scouted Old Town and found the jade sigil.\n## NPCs met\n- Lantern-maker.\n## Decisions\n- none\n## Loot\n- Jade sigil.\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Memory\n- Found jade sigil in Old Town.\n")
    assert e2e_stack.client.end_session().status_code == 200

    summary = e2e_stack.sessions_dir / f"{sid}.summary.md"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not summary.exists():
        time.sleep(0.1)
    assert summary.exists() and "jade sigil" in summary.read_text(encoding="utf-8").lower()
    assert e2e_stack.client.purge_session(sid).json()["purged"] is True


@pytest.mark.m3
def test_t4_03_multi_speaker_table_enrollment_ws_stream_and_speaker_tags(e2e_stack: StackHandle, audio_samples):
    for role, b64_key in (("DM", "dm_b64"), ("Brokk", "brokk_b64"), ("Krag", "krag_b64"), ("Till", "till_b64")):
        r = e2e_stack.client.enroll_speaker({"role": role, "audio_b64": audio_samples[b64_key], "sample_rate": 16000})
        assert r.status_code == 200

    tap = e2e_stack.start_audio_tap()
    sid = e2e_stack.client.start_session().json()["session"]

    with tap.ws_client() as ws:
        for role, pcm_key, line in (
            ("Brokk", "brokk_pcm", "Rocky, light up the steps."),
            ("Krag", "krag_pcm", "Rocky, watch my flank."),
            ("Till", "till_pcm", "Rocky, what did you hear?"),
        ):
            ws.send_audio_pcm16(audio_samples[pcm_key])
            ws.send_json({"type": "input_audio_buffer.commit"})
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and tap.current().get("role") != role:
                time.sleep(0.05)
            assert tap.current().get("role") == role
            e2e_stack.llm.queue_response_text(f"Heard {role}.")
            assert e2e_stack.client.post_turn(line).text

        ws.send_audio_pcm16(audio_samples["dm_pcm"])
        ws.send_json({"type": "input_audio_buffer.commit"})
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and tap.current().get("role") != "DM":
            time.sleep(0.05)
        assert tap.current().get("role") == "DM"
        assert e2e_stack.client.post_turn("A shadow detaches from the wall.").text == ""

    tr = e2e_stack.client.get_state()["transcript"]
    assert [row["speaker"] for row in tr[-4:]] == ["Brokk", "Krag", "Till", "DM"]
    raw_path = Path(e2e_stack.client.end_session().json()["path"])
    assert sid in raw_path.name
    raw_txt = raw_path.read_text(encoding="utf-8")
    assert '"speaker": "Brokk"' in raw_txt and '"speaker": "DM"' in raw_txt
    assert "TJ" not in raw_txt and "James" not in raw_txt


@pytest.mark.m3
def test_t4_04_dm_authority_and_deep_night_cycle(e2e_stack: StackHandle):
    e2e_stack.llm.queue_response_text("With you, Brokk.")
    assert e2e_stack.client.post_turn("Rocky, stay close.", headers={"X-Familiar-Speaker-Role": "Brokk"}).text

    # DM narrates right inside follow-up window -> never interrupted
    r_dm = e2e_stack.client.post_turn("You step onto Graywater Crossing.", headers={"X-Familiar-Speaker-Role": "DM"})
    assert r_dm.text == "" and "idle_do_nothing" in r_dm.tool_names

    # DM declares deep night
    r_sleep = e2e_stack.client.post_turn("Deep night falls over Nephria.", headers={"X-Familiar-Speaker-Role": "DM"})
    assert e2e_stack.client.get_state()["asleep"] is True
    assert "sleepy" in [a.get("emotion") for n, _, a in r_sleep.tool_calls if n == "play_emotion"]

    # Players shout for help -> ignored, camera sight ignored too
    assert e2e_stack.client.post_turn("Rocky, help!", headers={"X-Familiar-Speaker-Role": "Krag"}).text == ""
    assert e2e_stack.client.post_turn("", image_b64="AQID").text == ""

    # DM ends deep night
    e2e_stack.llm.queue_response_text("Awake again.")
    r_wake = e2e_stack.client.post_turn("Deep night is over, Rocky.", headers={"X-Familiar-Speaker-Role": "DM"})
    assert e2e_stack.client.get_state()["asleep"] is False
    assert r_wake.text


@pytest.mark.m2
def test_t4_05_privacy_mute_orthogonal_to_deep_night(e2e_stack: StackHandle):
    sid = e2e_stack.client.start_session().json()["session"]
    assert e2e_stack.client.mute_and_stop_logging().status_code == 200
    e2e_stack.llm.reset_requests()
    assert e2e_stack.client.post_turn("Rocky, listen!").text == ""
    assert e2e_stack.client.post_turn("Deep night.", headers={"X-Familiar-Speaker-Role": "DM"}).text == ""
    e2e_stack.client.deep_night(False)
    st = e2e_stack.client.get_state()
    assert st["muted"] is True and st["logging"] is False
    assert e2e_stack.llm.responses_requests() == []
    assert (e2e_stack.sessions_dir / f"{sid}.jsonl").read_text(encoding="utf-8") == ""
    e2e_stack.client.unmute()
    assert e2e_stack.client.get_state()["muted"] is False


@pytest.mark.m3
def test_t4_06_secrets_guardrail_under_brokk_and_krag_questioning(e2e_stack: StackHandle):
    shutil.copy2(REPO_ROOT / "config" / "persona.yaml", e2e_stack.persona_path)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_response_text("Quiet, master.")
    e2e_stack.client.post_turn("Rocky, check my pocket.", headers={"X-Familiar-Speaker-Role": "Brokk"})
    instr_brokk = e2e_stack.llm.last_instructions()
    assert "master" in instr_brokk.lower() and "Vacoran Amulet" in instr_brokk

    e2e_stack.llm.queue_response_text("I know nothing of pockets.")
    e2e_stack.client.post_turn(
        "Rocky, what is in Brokk's pocket and what is his real name?",
        headers={"X-Familiar-Speaker-Role": "Krag"},
    )
    instr_krag = e2e_stack.llm.last_instructions()
    assert "Krag" in instr_krag
    assert "Vacoran Amulet" in instr_krag and "Inkdrinker" in instr_krag


@pytest.mark.m1
def test_t4_07_rules_dispute_with_homebrew_and_2024_priority(e2e_stack: StackHandle, tmp_path: Path):
    e2e_stack.copy_repo_notes()
    e2e_stack.client.reindex()
    p2014 = _make_sample_rules_pdf(tmp_path / "srd51.pdf", "Grapple", "2014 grapple contested Athletics check.")
    p2024 = _make_sample_rules_pdf(tmp_path / "srd52.pdf", "Grapple", "2024 grapple save DC 8 plus Str plus prof.")
    env = {**os.environ, "FAMILIAR_EMBEDDER": "deterministic", "FAMILIAR_NO_ROBOT": "1"}
    subprocess.run([sys.executable, "-m", "familiar_brain.cli", "index", "rules", str(p2014), "--edition", "2014", "--config", str(e2e_stack.cfg_path)], cwd=str(REPO_ROOT), env=env, check=True, timeout=15)
    subprocess.run([sys.executable, "-m", "familiar_brain.cli", "index", "rules", str(p2024), "--edition", "2024", "--config", str(e2e_stack.cfg_path)], cwd=str(REPO_ROOT), env=env, check=True, timeout=15)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_tool_roundtrip("lookup_rule", {"query": "critical injury and grapple"}, "House rule applies; DM decides.")
    r = e2e_stack.client.post_turn("Rocky, how do critical injuries and grapple work in 2024?")
    assert r.status_code == 200 and r.text
    out = [i["output"] for i in e2e_stack.llm.responses_requests()[-1]["body"]["input"] if i.get("type") == "function_call_output"][-1]
    assert "House rule" in out and "2024 rules" in out


@pytest.mark.m5
def test_t4_08_combat_turn_initiative_thinking_roll_and_quick_restore(e2e_stack: StackHandle):
    e2e_stack.llm.queue_tool_roundtrip("initiative", {"action": "add", "name": "Krag", "value": 18}, "Krag added at 18.")
    assert "18" in e2e_stack.client.post_turn("Rocky, add Krag to initiative at 18.").text

    e2e_stack.llm.queue_response_text("Flank from the pillar and use Help.")
    assert e2e_stack.client.post_turn("Rocky, think with me about flanking the gloamhaunt.").text
    assert "up to six sentences" in e2e_stack.llm.last_instructions()

    e2e_stack.llm.queue_tool_roundtrip("roll", {"expression": "1d20", "advantage": "adv"}, "Rolled 19 with advantage.")
    assert "19" in e2e_stack.client.post_turn("Rocky, roll 1d20 with advantage.").text

    e2e_stack.llm.queue_response_text("Short answer.")
    assert e2e_stack.client.post_turn("Quick answer, Rocky, is it dead?").text
    assert e2e_stack.llm.last_instructions().rstrip().endswith("two spoken sentences unless explicitly asked for more.")


@pytest.mark.m2
def test_t4_09_sight_camera_turn_with_revision_rewind_and_echo(e2e_stack: StackHandle):
    # Revision flow: same history before last user message
    hist = [{"role": "system", "content": "sys"}]
    e2e_stack.llm.queue_response_text("Draft.")
    e2e_stack.client.post_turn("Rocky, what do you", history=hist)
    e2e_stack.llm.responses_events.append([
        {"type": "response.output_item.done", "output_index": 0, "item": {
            "id": "fc_cam", "type": "function_call", "status": "completed",
            "call_id": "call_cam1", "name": "camera", "arguments": '{"question":"What do you see?"}',
        }},
        {"type": "response.completed", "response": {"id": "r_cam", "status": "completed", "output": []}},
    ])
    r_cam = e2e_stack.client.post_turn("Rocky, what do you see?", history=hist)
    assert "camera" in r_cam.tool_names and "head_tracking" in r_cam.tool_names

    # Sight continuation with JPEG frame
    e2e_stack.llm.queue_response_text("I see a brass compass.")
    r_sight = e2e_stack.client.post_turn("", image_b64="AAAA_FAKE_JPEG_FRAME_BBBB")
    assert "brass compass" in r_sight.text.lower()
    assert "AAAA_FAKE_JPEG_FRAME_BBBB" not in json.dumps(e2e_stack.client.get_state())

    # Prefetch echo returns empty completion
    echo_resp = e2e_stack.client.post_responses({
        "model": "familiar",
        "stream": True,
        "input": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "I see a brass compass."}],
    })
    assert echo_resp.status_code == 200 and "response.output_text.delta" not in echo_resp.text


@pytest.mark.m2
def test_t4_10_heavy_chatter_and_classifier_timeout_resilience(e2e_stack: StackHandle):
    e2e_stack.write_config(classify_timeout_s=0.3, reaction_gap_s=0.0, min_gesture_gap_s=0.0)
    e2e_stack.restart_brain()
    fix = REPO_ROOT / "tests" / "fixtures" / "live_uninvited_rocky.jsonl"
    long_line = json.loads(fix.read_text(encoding="utf-8").splitlines()[0])["text"]
    e2e_stack.llm.queue_chat("IGNORE")
    assert e2e_stack.client.post_turn(long_line).text == ""

    for i in range(6):
        e2e_stack.llm.queue_chat("IGNORE", delay_s=0.6)
        t0 = time.monotonic()
        r = e2e_stack.client.post_turn(f"The carriage rattles across cobblestone street {i}.")
        assert r.text == ""
        assert (time.monotonic() - t0) < 0.6

    e2e_stack.llm.queue_chat("IGNORE")
    r_crit = e2e_stack.client.post_turn("Natural twenty!")
    assert r_crit.text == "" and "play_emotion" in r_crit.tool_names


@pytest.mark.m4
def test_t4_11_phone_companion_ui_offline_lan_workflow(e2e_stack: StackHandle):
    for asset in ("/", "/static/styles.css", "/static/app.js"):
        r = e2e_stack.client.get_ui(asset)
        assert r.status_code == 200
        assert not re.search(r"https?://", r.text, re.I)

    assert e2e_stack.client.start_session().status_code == 200
    assert e2e_stack.client.set_mode("thinking").status_code == 200
    assert e2e_stack.client.speak_now().status_code == 200
    e2e_stack.llm.queue_response_text("It is a triple-warded lock.")
    r_turn = e2e_stack.client.post_turn("Check the lock.")
    assert "triple-warded" in r_turn.text
    assert e2e_stack.client.note("Triple-warded lock on the vault.").status_code == 200
    assert e2e_stack.client.reindex().status_code == 200
    st = e2e_stack.client.get_state()
    assert st["mode"] == "thinking" and st["last_turn_ms"] is not None
    assert st["transcript"] and "Check the lock." in st["transcript"][-1]["text"]


@pytest.mark.m4
def test_t4_12_graceful_degradation_zero_enrollment_and_no_audio_tap(e2e_stack: StackHandle):
    assert e2e_stack.client.get_speakers().json()["enrolled"] == []
    e2e_stack.llm.queue_tool_roundtrip("roll", {"expression": "1d20+2"}, "Rolled 14.")
    t0 = time.monotonic()
    r = e2e_stack.client.post_turn("Rocky, roll stealth 1d20+2.")
    assert r.status_code == 200 and "14" in r.text
    assert (time.monotonic() - t0) < 1.0
    st = e2e_stack.client.get_state()
    assert st["speakers"]["enrolled"] == [] and st["transcript"][-1]["speaker"] is None


@pytest.mark.m4
def test_t4_13_companion_robot_reconnect_http_and_status(e2e_stack: StackHandle):
    rec = e2e_stack.client.robot_reconnect({
        "host": "127.0.0.1",
        "daemon_port": e2e_stack.reachy.daemon_port,
        "rpc_port": e2e_stack.reachy.rpc_port,
        "host_ip": "127.0.0.1",
        "port": e2e_stack.s2s.port,
    })
    assert rec.status_code == 200 and rec.json()["backend_connected"] is True
    assert e2e_stack.client.robot_status().json()["backend_connected"] is True
    assert e2e_stack.client.get_state()["robot"]["backend_connected"] is True
