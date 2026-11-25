"""Tier 1 E2E Tests: Happy-Path Behavior Across Features F1-F21 (105 tests).

Strictly opaque-box: exercises HTTP/SSE, WebSocket, files, and CLI subprocesses only.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pymupdf
import pytest
import requests

from tests.e2e.conftest import REPO_ROOT, StackHandle, to_bash_path
from tests.e2e.fakes import pcm16_to_b64

pytestmark = [pytest.mark.e2e, pytest.mark.t1]


# ---------------------------------------------------------------------------
# Shared opaque-box test helpers
# ---------------------------------------------------------------------------

def _has_bash() -> bool:
    return shutil.which("bash") is not None


def _has_powershell() -> str | None:
    if sys.platform == "win32":
        return shutil.which("powershell.exe") or shutil.which("powershell")
    return shutil.which("pwsh")


def _write_lf_script(path: Path, content: str) -> Path:
    path.write_text(content.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass
    return path


def _start_all_cmd(*args: str) -> list[str]:
    arg_str = (" " + " ".join(shlex.quote(a) for a in args)) if args else ""
    return [
        "bash",
        "-c",
        f'if [ -n "${{STUB_BIN_DIR:-}}" ]; then export PATH="$STUB_BIN_DIR:$PATH"; fi; exec bash scripts/start_all.sh{arg_str}',
    ]


def _setup_start_all_stubs(
    tmp_path: Path,
    *,
    fail_llama: bool = False,
    fail_s2s_pool: bool = False,
    fail_robot: bool = False,
    s2s_lifetime_s: float = 1.2,
) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "stub_bin"
    s2s_bin = tmp_path / "s2s_venv" / "bin"
    log_dir = tmp_path / "stub_logs"
    bin_dir.mkdir(parents=True, exist_ok=True)
    s2s_bin.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    trace_file = log_dir / "trace.log"
    trace_b = to_bash_path(trace_file)
    log_b = to_bash_path(log_dir)

    llama = _write_lf_script(
        bin_dir / "llama-server",
        f"""#!/usr/bin/env bash
echo "llama:$*" >> "{trace_b}"
{"exit 1" if fail_llama else f"touch '{log_b}/llama.ready'; sleep 5"}
""",
    )
    _write_lf_script(
        bin_dir / "uv",
        f"""#!/usr/bin/env bash
echo "uv:$*" >> "{trace_b}"
touch "{log_b}/brain.ready"
sleep 5
""",
    )
    _write_lf_script(
        s2s_bin / "speech-to-speech",
        f"""#!/usr/bin/env bash
echo "s2s:$*" >> "{trace_b}"
{"sleep 0.2" if fail_s2s_pool else f"touch '{log_b}/s2s.ready'; sleep {s2s_lifetime_s}"}
""",
    )
    _write_lf_script(
        bin_dir / "curl",
        f"""#!/usr/bin/env bash
echo "curl:$*" >> "{trace_b}"
case "$*" in
  *":8080/health"*)
    [ -f "{log_b}/llama.ready" ] && exit 0 || exit 1
    ;;
  *":8090/api/state"*)
    [ -f "{log_b}/brain.ready" ] && exit 0 || exit 1
    ;;
  *"/v1/pool"*)
    {"exit 1" if fail_s2s_pool else f'[ -f "{log_b}/s2s.ready" ] && exit 0 || exit 1'}
    ;;
  *"/api/session/start"*)
    echo '{{"session":"stub-sid","ended":null}}'
    exit 0
    ;;
  *)
    exit 0
    ;;
esac
""",
    )
    py_body = (
        'echo \'{"backend_connected": false}\' >&2; exit 1'
        if fail_robot
        else 'echo \'{"backend_connected": true}\'; exit 0'
    )
    _write_lf_script(
        bin_dir / "python3",
        f"""#!/usr/bin/env bash
echo "python3:$*" >> "{trace_b}"
{py_body}
""",
    )
    env = dict(os.environ)
    env["STUB_BIN_DIR"] = to_bash_path(bin_dir)
    env["LLAMA_BIN"] = to_bash_path(llama)
    env["S2S_VENV"] = to_bash_path(tmp_path / "s2s_venv")
    env["UV_PROJECT_ENVIRONMENT"] = to_bash_path(tmp_path / "brain_venv")
    if sys.platform == "win32":
        wsl_vars = [
            "STUB_BIN_DIR",
            "LLAMA_BIN",
            "S2S_VENV",
            "UV_PROJECT_ENVIRONMENT",
            "BRAIN_PY",
            "SKIP_ROBOT",
            "AUTO_SESSION",
            "ENABLE_AUDIO_TAP",
            "STARTUP_TIMEOUT_S",
        ]
        existing = [v for v in env.get("WSLENV", "").split(":") if v]
        for v in wsl_vars:
            if v not in existing:
                existing.append(v)
        env["WSLENV"] = ":".join(existing)
    else:
        env["PATH"] = f"{to_bash_path(bin_dir)}:{env.get('PATH', '')}"
    return env, trace_file


def _make_sample_rules_pdf(path: Path, title: str, body_text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), f"{title}\n\n{body_text}")
    doc.set_toc([[1, title, 1]])
    doc.save(str(path))
    doc.close()
    return path


def _make_openai_shim(tmp_path: Path) -> Path:
    """Create a wire-level openai Responses API client shim ONLY when openai isn't installed in .venv."""
    pkg = tmp_path / "openai_shim" / "openai"
    types_pkg = pkg / "types" / "responses"
    types_pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        """
import json, types, requests

class _Resp:
    def __init__(self, d):
        self.id = d.get("id", "")
        self.status = d.get("status", "")
        self.output = d.get("output", [])
_Resp.__name__ = "Response"

class _ResponsesAPI:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
    def create(self, *, model="familiar", input=None, stream=False, timeout=30, **kw):
        body = {"model": model, "input": input, **kw}
        if stream:
            body["stream"] = True
        r = requests.post(f"{self.base_url}/responses", json=body, stream=bool(stream), timeout=timeout)
        r.raise_for_status()
        if not stream:
            if "application/json" not in r.headers.get("Content-Type", ""):
                raise ValueError("Expected JSON response")
            return _Resp(r.json())
        from openai.types.responses import (
            ResponseCompletedEvent, ResponseFunctionToolCall,
            ResponseOutputItemDoneEvent, ResponseOutputMessage, ResponseTextDeltaEvent,
        )
        events = []
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            t = ev.get("type")
            if t == "response.output_text.delta":
                events.append(ResponseTextDeltaEvent(ev.get("delta", "")))
            elif t == "response.output_item.done":
                item = ev.get("item") or {}
                if item.get("type") == "function_call":
                    obj = ResponseFunctionToolCall(item.get("name", ""), item.get("call_id", ""), item.get("arguments", "{}"))
                else:
                    obj = ResponseOutputMessage()
                events.append(ResponseOutputItemDoneEvent(obj))
            elif t == "response.completed":
                events.append(ResponseCompletedEvent())
        return iter(events)

class OpenAI:
    def __init__(self, *, base_url, api_key=""):
        self.responses = _ResponsesAPI(base_url)
""",
        encoding="utf-8",
    )
    (pkg / "types" / "__init__.py").write_text("", encoding="utf-8")
    (types_pkg / "__init__.py").write_text(
        """
class ResponseCompletedEvent: pass
class ResponseOutputMessage: pass
class ResponseTextDeltaEvent:
    def __init__(self, delta): self.delta = delta
class ResponseFunctionToolCall:
    def __init__(self, name, call_id, arguments):
        self.name, self.call_id, self.arguments = name, call_id, arguments
class ResponseOutputItemDoneEvent:
    def __init__(self, item): self.item = item
""",
        encoding="utf-8",
    )
    return tmp_path / "openai_shim"


def _run_sdk_probe(base_url: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["FAMILIAR_NO_ROBOT"] = "1"
    py = sys.executable
    if importlib.util.find_spec("openai") is None:
        s2s_py = Path.home() / ".venvs" / "s2s" / "bin" / "python"
        if s2s_py.exists():
            py = str(s2s_py)
        else:
            shim_dir = _make_openai_shim(tmp_path)
            env["PYTHONPATH"] = str(shim_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [py, str(REPO_ROOT / "scripts" / "sdk_contract_probe.py"), "--base-url", base_url],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


# ===========================================================================
# F1: Self-contained start_all.sh (m5)
# ===========================================================================

@pytest.mark.m5
def test_t1_f01_1_start_all_help_documents_flags():
    if not _has_bash():
        pytest.skip("bash not available")
    proc = subprocess.run(
        ["bash", "scripts/start_all.sh", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout + proc.stderr
    for flag in ("--skip-robot", "--auto-session", "--no-audio-tap"):
        assert flag in out


@pytest.mark.m5
def test_t1_f01_2_start_all_shell_syntax_valid():
    if not _has_bash():
        pytest.skip("bash not available")
    proc = subprocess.run(
        ["bash", "-n", "scripts/start_all.sh"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.m5
def test_t1_f01_3_start_all_ordered_startup_and_readiness(tmp_path: Path):
    if not _has_bash():
        pytest.skip("bash not available")
    env, trace_file = _setup_start_all_stubs(tmp_path, s2s_lifetime_s=0.8)
    proc = subprocess.run(
        _start_all_cmd("--skip-robot"),
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    trace = trace_file.read_text(encoding="utf-8")
    assert "llama:" in trace and "uv:" in trace and "s2s:" in trace
    assert trace.index("llama:") < trace.index("uv:") < trace.index("s2s:")
    assert ":8080/health" in trace and ":8090/api/state" in trace and "/v1/pool" in trace
    assert proc.returncode == 0 or "Point the robot at" in proc.stdout


@pytest.mark.m5
def test_t1_f01_4_start_all_auto_session_calls_session_start(tmp_path: Path):
    if not _has_bash():
        pytest.skip("bash not available")
    env, trace_file = _setup_start_all_stubs(tmp_path, s2s_lifetime_s=0.8)
    subprocess.run(
        _start_all_cmd("--skip-robot", "--auto-session"),
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    trace = trace_file.read_text(encoding="utf-8")
    assert "/api/session/start" in trace


@pytest.mark.m5
def test_t1_f01_5_start_all_clean_shutdown_on_sigterm(tmp_path: Path):
    if not _has_bash():
        pytest.skip("bash not available")
    env, trace_file = _setup_start_all_stubs(tmp_path, s2s_lifetime_s=20.0)
    proc = subprocess.Popen(
        _start_all_cmd("--skip-robot"),
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if trace_file.exists() and "s2s:" in trace_file.read_text(encoding="utf-8"):
                break
            time.sleep(0.1)
        assert trace_file.exists() and "s2s:" in trace_file.read_text(encoding="utf-8")
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=6.0)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)


# ===========================================================================
# F2: Robust robot_connect.py (m5)
# ===========================================================================

@pytest.mark.m5
def test_t1_f02_1_robot_connect_discovers_and_connects_fake_reachy(fake_reachy, fake_s2s):
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "robot_connect.py"),
            "--robot", "127.0.0.1",
            "--daemon-port", str(fake_reachy.daemon_port),
            "--rpc-port", str(fake_reachy.rpc_port),
            "--host-ip", "127.0.0.1",
            "--port", str(fake_s2s.port),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert any(c[1] == "/api/apps/current-app-status" for c in fake_reachy.daemon_calls)
    assert any(c["method"] == "conversation.status" for c in fake_reachy.rpc_calls)


@pytest.mark.m5
def test_t1_f02_2_robot_connect_starts_stopped_app_and_configures_backend(fake_reachy, fake_s2s):
    fake_reachy.app_running = False
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "robot_connect.py"),
            "--robot", "127.0.0.1",
            "--daemon-port", str(fake_reachy.daemon_port),
            "--rpc-port", str(fake_reachy.rpc_port),
            "--host-ip", "127.0.0.1",
            "--port", str(fake_s2s.port),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert ("POST", "/api/apps/start-app/reachy_mini_conversation_app") in fake_reachy.daemon_calls
    assert fake_reachy.configured_host == "127.0.0.1"
    assert fake_reachy.configured_port == fake_s2s.port


@pytest.mark.m5
def test_t1_f02_3_robot_connect_applies_familiar_profile_tools_and_voice(fake_reachy, fake_s2s):
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "robot_connect.py"),
            "--robot", "127.0.0.1",
            "--daemon-port", str(fake_reachy.daemon_port),
            "--rpc-port", str(fake_reachy.rpc_port),
            "--host-ip", "127.0.0.1",
            "--port", str(fake_s2s.port),
            "--voice", "Aiden",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert fake_reachy.applied_personality == "familiar"
    assert fake_reachy.applied_voice == "Aiden"
    assert "play_emotion" in fake_reachy.enabled_tools
    apply_calls = [c for c in fake_reachy.rpc_calls if c["method"] == "personalities.apply"]
    assert apply_calls and apply_calls[-1]["params"].get("persist") is True


@pytest.mark.m5
def test_t1_f02_4_robot_connect_polls_until_backend_connected_true(fake_reachy, fake_s2s):
    fake_reachy.connect_after_polls = 2
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "robot_connect.py"),
            "--robot", "127.0.0.1",
            "--daemon-port", str(fake_reachy.daemon_port),
            "--rpc-port", str(fake_reachy.rpc_port),
            "--host-ip", "127.0.0.1",
            "--port", str(fake_s2s.port),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert fake_reachy.backend_connected is True
    assert '"backend_connected": true' in proc.stdout.lower()


@pytest.mark.m5
def test_t1_f02_5_robot_connect_dry_run_is_strictly_read_only(fake_reachy, fake_s2s):
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "robot_connect.py"),
            "--robot", "127.0.0.1",
            "--daemon-port", str(fake_reachy.daemon_port),
            "--rpc-port", str(fake_reachy.rpc_port),
            "--host-ip", "127.0.0.1",
            "--port", str(fake_s2s.port),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    methods = {c["method"] for c in fake_reachy.rpc_calls}
    assert {"conversation.status", "personalities.list", "voices.list", "profile_tools.get"} <= methods
    assert not methods & {"backend.config", "personalities.save", "personalities.apply", "profile_tools.save", "voices.apply"}


# ===========================================================================
# F3: Windows Launcher launch.ps1 + launch_rocky.bat (m5)
# ===========================================================================

@pytest.mark.m5
def test_t1_f03_1_launch_bat_exists_and_invokes_launch_ps1():
    bat = REPO_ROOT / "launch_rocky.bat"
    assert bat.exists(), "launch_rocky.bat must exist at repo root"
    txt = bat.read_text(encoding="utf-8")
    assert "launch.ps1" in txt and "powershell" in txt.lower()


@pytest.mark.m5
def test_t1_f03_2_launch_ps1_dry_run_prints_preflight_and_ui_url():
    ps = _has_powershell()
    if not ps:
        pytest.skip("PowerShell not available")
    proc = subprocess.run(
        [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(REPO_ROOT / "scripts" / "launch.ps1"), "-DryRun", "-SkipRobot"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = proc.stdout
    assert "8765" in out and "8090" in out
    assert re.search(r"http://[^\s:]+:8090", out)


@pytest.mark.m5
def test_t1_f03_3_launch_ps1_cleans_stale_ssl_cert_file():
    ps = _has_powershell()
    if not ps:
        pytest.skip("PowerShell not available")
    env = dict(os.environ)
    env["SSL_CERT_FILE"] = r"C:\ProgramData\miniconda3\ssl\cacert.pem"
    proc = subprocess.run(
        [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(REPO_ROOT / "scripts" / "launch.ps1"), "-DryRun", "-SkipRobot"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "SSL_CERT_FILE" in proc.stdout


@pytest.mark.m5
def test_t1_f03_4_launch_ps1_dry_run_skip_robot_handles_unreachable_host():
    ps = _has_powershell()
    if not ps:
        pytest.skip("PowerShell not available")
    proc = subprocess.run(
        [
            ps, "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(REPO_ROOT / "scripts" / "launch.ps1"),
            "-DryRun", "-SkipRobot", "-RobotHost", "nonexistent.invalid.local",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


@pytest.mark.m5
def test_t1_f03_5_launch_ps1_forwards_auto_session_and_no_audio_tap():
    ps = _has_powershell()
    if not ps:
        pytest.skip("PowerShell not available")
    proc = subprocess.run(
        [
            ps, "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(REPO_ROOT / "scripts" / "launch.ps1"),
            "-DryRun", "-SkipRobot", "-AutoSession", "-NoAudioTap",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "--auto-session" in proc.stdout and "--no-audio-tap" in proc.stdout


# ===========================================================================
# F4: STT Runaway Turn Fix (m5)
# ===========================================================================

@pytest.mark.m5
def test_t1_f04_1_start_all_passes_bounded_stt_flags():
    txt = (REPO_ROOT / "scripts" / "start_all.sh").read_text(encoding="utf-8")
    assert "--max_speech_ms" in txt
    assert "--unanswered_reopen_ms" in txt
    assert "--smart_turn_max_wait_ms" in txt


@pytest.mark.m5
def test_t1_f04_2_stt_flag_values_bounded_in_start_all():
    txt = (REPO_ROOT / "scripts" / "start_all.sh").read_text(encoding="utf-8")
    m_reopen = re.search(r"--unanswered_reopen_ms\s+(\d+)", txt)
    m_max = re.search(r"--max_speech_ms\s+(\d+)", txt)
    assert m_reopen and int(m_reopen.group(1)) < 7000
    assert m_max and 1000 <= int(m_max.group(1)) <= 60000


@pytest.mark.m5
def test_t1_f04_3_rapid_turn_revisions_preserve_preface_without_duplication(e2e_stack: StackHandle):
    e2e_stack.llm.queue_response_text("First draft.")
    e2e_stack.llm.queue_response_text("Revised reply.")
    history = [{"role": "system", "content": "placeholder"}]
    r1 = e2e_stack.client.post_turn("Rocky, roll", history=history)
    r2 = e2e_stack.client.post_turn("Rocky, roll me a d20", history=history)
    assert r1.status_code == 200 and r2.status_code == 200
    assert "head_tracking" in r1.tool_names and "head_tracking" in r2.tool_names


@pytest.mark.m3
def test_t1_f04_4_continuous_long_audio_stream_keeps_tap_bounded(e2e_stack: StackHandle, audio_samples):
    tap = e2e_stack.start_audio_tap()
    chunk = audio_samples["brokk_pcm"]  # 2s PCM16 = 64KB
    with tap.ws_client() as ws:
        ws.send_json({"type": "session.update", "session": {"input_audio_format": "pcm16"}})
        # Stream 60s of simulated continuous speech rapidly (30 x 2s bursts)
        for _ in range(30):
            ws.send_audio_pcm16(chunk, chunk_bytes=16000)
        pong = ws.ping(b"alive")
        assert pong == b"alive"
    # Verify tap control still responsive and frames relayed
    cur = tap.current()
    assert "capturing" in cur
    assert len(e2e_stack.s2s.ws_messages) > 30


@pytest.mark.m5
def test_t1_f04_5_versions_md_documents_stt_bounds_and_stress_test():
    txt = (REPO_ROOT / "docs" / "versions.md").read_text(encoding="utf-8")
    assert "max_speech_ms" in txt and "unanswered_reopen_ms" in txt


# ===========================================================================
# F5: Gemma 4 E4B-it QAT Upgrade (m5)
# ===========================================================================

@pytest.mark.m5
def test_t1_f05_1_start_all_configures_gemma4_qat_and_kv_quant():
    txt = (REPO_ROOT / "scripts" / "start_all.sh").read_text(encoding="utf-8")
    assert "gemma-4-E4B-it" in txt
    assert "--jinja" in txt and "-np 2" in txt and "--mmproj-auto" in txt
    assert "-ctk" in txt and "-ctv" in txt


@pytest.mark.m5
def test_t1_f05_2_brain_sends_reasoning_none_and_flat_tools(e2e_stack: StackHandle):
    e2e_stack.llm.queue_response_text("Standing by.")
    r = e2e_stack.client.post_turn("Rocky, hello")
    assert r.status_code == 200
    reqs = e2e_stack.llm.responses_requests()
    assert reqs
    body = reqs[-1]["body"]
    assert body.get("reasoning") == {"effort": "none"}
    tool_names = {t.get("name") for t in body.get("tools", [])}
    assert {"roll", "lookup_rule", "recall", "note", "initiative"} <= tool_names


@pytest.mark.m5
def test_t1_f05_3_roll_tool_roundtrip_hides_tool_from_s2s_client(e2e_stack: StackHandle):
    e2e_stack.llm.queue_tool_roundtrip("roll", {"expression": "1d20+3"}, "You rolled a natural twenty!")
    r = e2e_stack.client.post_turn("Rocky, roll 1d20+3")
    assert r.status_code == 200
    assert "roll" not in r.tool_names
    assert "natural twenty" in r.text.lower()
    reqs = e2e_stack.llm.responses_requests()
    assert len(reqs) == 2
    outputs = [i for i in reqs[-1]["body"].get("input", []) if i.get("type") == "function_call_output"]
    assert outputs and "total" in outputs[-1]["output"].lower()


@pytest.mark.m5
def test_t1_f05_4_sight_turn_attaches_input_image_upstream(e2e_stack: StackHandle):
    e2e_stack.llm.queue_response_text("I see a brass coin on the table.")
    fake_jpg = "AQIDBA=="
    r = e2e_stack.client.post_turn("", image_b64=fake_jpg)
    assert r.status_code == 200
    assert "brass coin" in r.text.lower()
    reqs = e2e_stack.llm.responses_requests()
    assert reqs
    serialized = json.dumps(reqs[-1]["body"])
    assert "input_image" in serialized and fake_jpg in serialized


@pytest.mark.m5
def test_t1_f05_5_versions_md_records_gemma4_qat_measurements():
    txt = (REPO_ROOT / "docs" / "versions.md").read_text(encoding="utf-8")
    assert "gemma-4-E4B-it" in txt and ("QAT" in txt or "Q4_0" in txt)


# ===========================================================================
# F6: TTS Benchmark & VRAM Headroom (m5)
# ===========================================================================

@pytest.mark.m5
def test_t1_f06_1_start_all_configures_selected_tts_backend():
    txt = (REPO_ROOT / "scripts" / "start_all.sh").read_text(encoding="utf-8")
    assert "--tts" in txt and ("qwen3" in txt or "kokoro" in txt)


@pytest.mark.m5
def test_t1_f06_2_versions_md_records_vram_benchmark_with_headroom():
    txt = (REPO_ROOT / "docs" / "versions.md").read_text(encoding="utf-8")
    assert ("Kokoro" in txt or "kokoro" in txt) and ("Qwen3" in txt or "qwen3" in txt)
    assert "VRAM" in txt


@pytest.mark.m5
def test_t1_f06_3_versions_md_records_tts_latencies_and_rationale():
    txt = (REPO_ROOT / "docs" / "versions.md").read_text(encoding="utf-8").lower()
    assert "tts" in txt and ("median" in txt or "latency" in txt)


@pytest.mark.m5
def test_t1_f06_4_robot_profile_voice_matches_tts_speaker():
    profile = (REPO_ROOT / "robot_profile" / "familiar" / "profile.md").read_text(encoding="utf-8")
    persona = (REPO_ROOT / "config" / "persona.example.yaml").read_text(encoding="utf-8")
    assert 'voice = "' in profile and "tts_speaker:" in persona


@pytest.mark.m5
def test_t1_f06_5_sdk_contract_probe_passes_against_running_brain(e2e_stack: StackHandle, tmp_path: Path):
    proc = _run_sdk_probe(f"{e2e_stack.brain_url}/v1", tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS:" in proc.stdout


# ===========================================================================
# F7: Unaddressed Turn Stall Elimination (m2)
# ===========================================================================

@pytest.mark.m2
def test_t1_f07_1_unaddressed_nonmatching_returns_idle_fast(e2e_stack: StackHandle):
    e2e_stack.llm.queue_chat("IGNORE")
    e2e_stack.client.post_turn("Setup line.")  # flush turn-1 head_tracking preface
    e2e_stack.llm.queue_chat("IGNORE")
    t0 = time.monotonic()
    r = e2e_stack.client.post_turn("Someone passes the pretzels across the table.")
    elapsed = time.monotonic() - t0
    assert r.status_code == 200
    assert "idle_do_nothing" in r.tool_names
    assert r.text == ""
    assert elapsed < 0.8


@pytest.mark.m2
def test_t1_f07_2_table_rules_emit_emotion_with_zero_reaction_llm_calls(e2e_stack: StackHandle):
    e2e_stack.llm.queue_chat("IGNORE")
    e2e_stack.llm.reset_requests()
    r = e2e_stack.client.post_turn("Natural twenty!")
    assert r.status_code == 200
    assert "play_emotion" in r.tool_names
    assert r.text == ""
    # Only gate classify call (if any), zero reaction calls
    assert len(e2e_stack.llm.chat_requests()) <= 1


@pytest.mark.m2
def test_t1_f07_3_classify_timeout_does_not_chain_second_reaction_timeout(e2e_stack: StackHandle):
    e2e_stack.write_config(classify_timeout_s=0.35)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_chat("IGNORE", delay_s=0.9)
    e2e_stack.llm.queue_chat("amused", delay_s=0.9)
    t0 = time.monotonic()
    r = e2e_stack.client.post_turn("The shadowy merchant opens the silver chest carefully.")
    elapsed = time.monotonic() - t0
    assert r.status_code == 200
    assert r.text == ""
    # Must NEVER pay 2 * classify_timeout_s (would be ~0.7-1.8s+)
    assert elapsed < 0.65, f"Double timeout stall detected: {elapsed:.2f}s"
    assert len(e2e_stack.llm.chat_requests()) <= 1


@pytest.mark.m2
def test_t1_f07_4_addressed_turn_with_roll_completes_under_1500ms(e2e_stack: StackHandle):
    e2e_stack.llm.queue_tool_roundtrip("roll", {"expression": "1d20"}, "Rolled 17.")
    t0 = time.monotonic()
    r = e2e_stack.client.post_turn("Rocky, roll 1d20")
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    assert r.status_code == 200
    assert "17" in r.text
    assert elapsed_ms < 1500.0


@pytest.mark.m2
def test_t1_f07_5_api_state_records_last_turn_ms(e2e_stack: StackHandle):
    e2e_stack.llm.queue_response_text("Ready.")
    e2e_stack.client.post_turn("Rocky, status?")
    st = e2e_stack.client.get_state()
    assert "last_turn_ms" in st
    assert isinstance(st["last_turn_ms"], (int, float)) and st["last_turn_ms"] >= 0.0


# ===========================================================================
# F8: nephria-update/ Notes & Reindex (m1)
# ===========================================================================

@pytest.mark.m1
def test_t1_f08_1_repo_notes_contain_all_nephria_update_files():
    notes = REPO_ROOT / "data" / "notes"
    for rel in (
        "10-brokk-trenevain.md",
        "11-roster-additions.md",
        "12-recap-old-town-and-jade-halls.md",
        "homebrew/rocky-familiar.md",
        "homebrew/brokk-sheet.md",
        "homebrew/house-rules-2026-10-02.md",
    ):
        assert (notes / rel).exists(), f"Missing {rel} in data/notes/"


@pytest.mark.m1
def test_t1_f08_2_docs_rocky_persona_facts_and_notes_policy_updated():
    facts = REPO_ROOT / "docs" / "rocky-persona-facts.md"
    policy = REPO_ROOT / "docs" / "notes-policy.md"
    assert facts.exists() and "Brokk Trenevain" in facts.read_text(encoding="utf-8")
    ptxt = policy.read_text(encoding="utf-8")
    assert "nephria-update" in ptxt and "SRD_CC_v5.2.1.pdf" in ptxt


@pytest.mark.m1
def test_t1_f08_3_reindex_nephria_notes_returns_indexed_count_and_enables_recall(e2e_stack: StackHandle):
    e2e_stack.copy_repo_notes()
    res = e2e_stack.client.reindex()
    assert res.status_code == 200
    data = res.json()
    assert data["indexed"] >= 6
    e2e_stack.llm.queue_tool_roundtrip("recall", {"query": "Brokk Trenevain", "scope": "notes"}, "Brokk is a bard and warlock.")
    r = e2e_stack.client.post_turn("Rocky, who is Brokk Trenevain?")
    assert r.status_code == 200
    reqs = e2e_stack.llm.responses_requests()
    tool_out = [i["output"] for i in reqs[-1]["body"]["input"] if i.get("type") == "function_call_output"]
    assert tool_out and "Brokk" in tool_out[-1]


@pytest.mark.m1
def test_t1_f08_4_recall_finds_brokk_trenevain_canon_and_sheet(e2e_stack: StackHandle):
    e2e_stack.copy_repo_notes()
    e2e_stack.client.reindex()
    e2e_stack.llm.queue_tool_roundtrip("recall", {"query": "Brokk Trenevain Gloamhaunt", "scope": "all"}, "He is level 5.")
    r = e2e_stack.client.post_turn("Rocky, tell me about Brokk Trenevain.")
    assert r.status_code == 200
    instr = e2e_stack.llm.last_instructions()
    assert "Brokk" in instr


@pytest.mark.m1
def test_t1_f08_5_recall_finds_jade_halls_and_old_town_recap(e2e_stack: StackHandle):
    e2e_stack.copy_repo_notes()
    e2e_stack.client.reindex()
    e2e_stack.llm.queue_tool_roundtrip("recall", {"query": "Jade Halls Old Town", "scope": "notes"}, "We visited the Jade Halls.")
    r = e2e_stack.client.post_turn("Rocky, what happened in the Jade Halls and Old Town?")
    assert r.status_code == 200
    reqs = e2e_stack.llm.responses_requests()
    tool_out = [i["output"] for i in reqs[-1]["body"]["input"] if i.get("type") == "function_call_output"]
    assert tool_out and ("Jade Halls" in tool_out[-1] or "Old Town" in tool_out[-1])


# ===========================================================================
# F9: Lexicon Additions Merge (m1)
# ===========================================================================

@pytest.mark.m1
def test_t1_f09_1_repo_lexicon_contains_canonical_nephria_mappings():
    txt = (REPO_ROOT / "config" / "lexicon.tsv").read_text(encoding="utf-8")
    canonicals = {
        line.split("\t", 1)[0].strip()
        for line in txt.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for term in ("Brokk", "Brokk Trenevain", "Trenevain", "Krag", "Caldus", "Nephria", "Till", "Jade Halls", "Gloamhaunt", "Vacoran Amulet", "Inkdrinker"):
        assert term in canonicals, f"Missing canonical term {term!r} in config/lexicon.tsv"


@pytest.mark.m1
def test_t1_f09_2_lexicon_rewrites_heard_forms_in_session_log_and_upstream(e2e_stack: StackHandle):
    shutil.copy2(REPO_ROOT / "config" / "lexicon.tsv", e2e_stack.lexicon_path)
    e2e_stack.restart_brain()
    e2e_stack.client.start_session()
    e2e_stack.llm.queue_response_text("Heard.")
    r = e2e_stack.client.post_turn("Rocky, Brock Trinivane went to jade halls with Craig")
    assert r.status_code == 200
    body = e2e_stack.llm.responses_requests()[-1]["body"]
    user_txt = json.dumps(body["input"])
    assert "Brokk Trenevain" in user_txt and "Jade Halls" in user_txt and "Krag" in user_txt
    end_r = e2e_stack.client.end_session()
    raw_path = Path(end_r.json()["path"])
    log_txt = raw_path.read_text(encoding="utf-8")
    assert "Brokk Trenevain" in log_txt and "Jade Halls" in log_txt and "Krag" in log_txt


@pytest.mark.m1
def test_t1_f09_3_lexicon_rewrites_nepheria_and_calvis(e2e_stack: StackHandle):
    shutil.copy2(REPO_ROOT / "config" / "lexicon.tsv", e2e_stack.lexicon_path)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_response_text("Checking.")
    r = e2e_stack.client.post_turn("Rocky, what did Nepheria and Calvis do?")
    assert r.status_code == 200
    user_txt = json.dumps(e2e_stack.llm.responses_requests()[-1]["body"]["input"])
    assert "Nephria" in user_txt and "Caldus" in user_txt


@pytest.mark.m1
def test_t1_f09_4_lexicon_and_aliases_address_rockie_and_rocki(e2e_stack: StackHandle):
    e2e_stack.llm.queue_response_text("Yes?")
    e2e_stack.llm.queue_response_text("Listening.")
    r1 = e2e_stack.client.post_turn("Rockie, roll 1d20")
    r2 = e2e_stack.client.post_turn("Rocki, hello")
    assert "idle_do_nothing" not in r1.tool_names and r1.text
    assert "idle_do_nothing" not in r2.tool_names and r2.text


@pytest.mark.m1
def test_t1_f09_5_lexicon_collision_entries_remain_commented_and_inactive(e2e_stack: StackHandle):
    txt = (REPO_ROOT / "config" / "lexicon.tsv").read_text(encoding="utf-8")
    for guarded in ("ever heard", "called us", "carry", "curry", "jailhouse", "software"):
        active_lines = [
            ln for ln in txt.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#") and guarded in ln.lower()
        ]
        assert not active_lines, f"Collision phrase {guarded!r} must stay commented out"
    shutil.copy2(REPO_ROOT / "config" / "lexicon.tsv", e2e_stack.lexicon_path)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_response_text("Okay.")
    e2e_stack.client.post_turn("Rocky, have you ever heard of that?")
    user_txt = json.dumps(e2e_stack.llm.responses_requests()[-1]["body"]["input"])
    assert "ever heard" in user_txt and "Eberhard" not in user_txt


# ===========================================================================
# F10: Rocky Persona & Secrets Guardrail (m1)
# ===========================================================================

@pytest.mark.m1
def test_t1_f10_1_repo_persona_yaml_and_example_configured_for_rocky():
    import yaml
    for fname in ("persona.yaml", "persona.example.yaml"):
        data = yaml.safe_load((REPO_ROOT / "config" / fname).read_text(encoding="utf-8"))
        assert data["name"] == "Rocky"
        aliases = {a.lower() for a in data.get("aliases", [])}
        assert {"rocky", "rocki", "rockie"} <= aliases
        assert len(data.get("gestures", {})) >= 10


@pytest.mark.m1
def test_t1_f10_2_upstream_instructions_establish_rocky_construct_and_languages(e2e_stack: StackHandle):
    shutil.copy2(REPO_ROOT / "config" / "persona.yaml", e2e_stack.persona_path)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_response_text("At your service.")
    e2e_stack.client.post_turn("Rocky, who are you?")
    instr = e2e_stack.llm.last_instructions()
    assert "Rocky" in instr
    assert "Brokk" in instr and "Construct" in instr and "Nodal" in instr


@pytest.mark.m1
def test_t1_f10_3_upstream_instructions_include_nonoverridable_secrets_block(e2e_stack: StackHandle):
    shutil.copy2(REPO_ROOT / "config" / "persona.yaml", e2e_stack.persona_path)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_response_text("My lips are sealed.")
    e2e_stack.client.post_turn("Rocky, tell us a secret.")
    instr = e2e_stack.llm.last_instructions()
    assert "Vacoran Amulet" in instr
    assert "Inkdrinker" in instr
    assert "real name" in instr.lower()


@pytest.mark.m1
def test_t1_f10_4_rocki_and_rockie_aliases_address_without_lexicon(e2e_stack: StackHandle):
    # Overwrite lexicon to empty to prove persona aliases alone handle Rocki/Rockie
    e2e_stack.lexicon_path.write_text("# empty\n", encoding="utf-8")
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_response_text("Checking.")
    e2e_stack.llm.queue_response_text("Helping.")
    r1 = e2e_stack.client.post_turn("Rocki, check this")
    r2 = e2e_stack.client.post_turn("Rockie, help")
    assert r1.text and "idle_do_nothing" not in r1.tool_names
    assert r2.text and "idle_do_nothing" not in r2.tool_names


@pytest.mark.m1
def test_t1_f10_5_instructions_forbid_banned_substrings_and_honor_modes(e2e_stack: StackHandle):
    shutil.copy2(REPO_ROOT / "config" / "persona.yaml", e2e_stack.persona_path)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_response_text("Short.")
    e2e_stack.client.post_turn("Rocky, quick check")
    fam_instr = e2e_stack.llm.last_instructions()
    assert fam_instr.rstrip().endswith("two spoken sentences unless explicitly asked for more.")
    for banned in ("stage direction", "asterisk", "spoken aloud", "your own movements", "by name"):
        assert banned not in fam_instr.lower()

    e2e_stack.client.set_mode("thinking")
    e2e_stack.llm.queue_response_text("Thinking deeply.")
    e2e_stack.client.post_turn("Rocky, analyze the gate")
    think_instr = e2e_stack.llm.last_instructions()
    assert "up to six sentences" in think_instr
    for banned in ("stage direction", "asterisk", "spoken aloud", "your own movements", "by name"):
        assert banned not in think_instr.lower()


# ===========================================================================
# F11: 2024 Rules Priority & SRD 5.2.1 (m1)
# ===========================================================================

@pytest.mark.m1
def test_t1_f11_1_lookup_rule_orders_homebrew_then_2024_then_2014(e2e_stack: StackHandle, tmp_path: Path):
    (e2e_stack.homebrew_dir / "grapple-house.md").write_text("# Grapple\nGrapple check house rule at our table.\n", encoding="utf-8")
    e2e_stack.client.reindex()
    p2014 = _make_sample_rules_pdf(tmp_path / "srd51.pdf", "Grapple", "2014 grapple athletics contest rule.")
    p2024 = _make_sample_rules_pdf(tmp_path / "srd52.pdf", "Grapple", "2024 grapple unarmed strike saving throw rule.")
    env = {**os.environ, "FAMILIAR_EMBEDDER": "deterministic", "FAMILIAR_NO_ROBOT": "1"}
    for pdf, ed in ((p2014, "2014"), (p2024, "2024")):
        subprocess.run(
            [sys.executable, "-m", "familiar_brain.cli", "index", "rules", str(pdf), "--edition", ed, "--config", str(e2e_stack.cfg_path)],
            cwd=str(REPO_ROOT), env=env, check=True, timeout=15,
        )
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_tool_roundtrip("lookup_rule", {"query": "grapple rule"}, "House rule applies first.")
    e2e_stack.client.post_turn("Rocky, how does grapple work?")
    reqs = e2e_stack.llm.responses_requests()
    tool_out = [i["output"] for i in reqs[-1]["body"]["input"] if i.get("type") == "function_call_output"][-1]
    assert "House rule" in tool_out and "2024 rules" in tool_out and "2014 rules" in tool_out
    assert tool_out.index("House rule") < tool_out.index("2024 rules") < tool_out.index("2014 rules")


@pytest.mark.m1
def test_t1_f11_2_lookup_rule_returns_2024_cleanly_without_homebrew(e2e_stack: StackHandle, tmp_path: Path):
    p2024 = _make_sample_rules_pdf(tmp_path / "srd52.pdf", "Study Action", "2024 Study action lets you make an Intelligence check.")
    env = {**os.environ, "FAMILIAR_EMBEDDER": "deterministic", "FAMILIAR_NO_ROBOT": "1"}
    subprocess.run(
        [sys.executable, "-m", "familiar_brain.cli", "index", "rules", str(p2024), "--edition", "2024", "--config", str(e2e_stack.cfg_path)],
        cwd=str(REPO_ROOT), env=env, check=True, timeout=15,
    )
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_tool_roundtrip("lookup_rule", {"query": "Study Action"}, "It uses an Intelligence check.")
    e2e_stack.client.post_turn("Rocky, what is the Study action?")
    tool_out = [i["output"] for i in e2e_stack.llm.responses_requests()[-1]["body"]["input"] if i.get("type") == "function_call_output"][-1]
    assert "2024 rules" in tool_out and "House rule" not in tool_out


@pytest.mark.m1
def test_t1_f11_3_lookup_rule_falls_back_to_2014_when_no_2024_match(e2e_stack: StackHandle, tmp_path: Path):
    p2014 = _make_sample_rules_pdf(tmp_path / "srd51.pdf", "Old Exhaustion", "2014 exhaustion six levels table.")
    env = {**os.environ, "FAMILIAR_EMBEDDER": "deterministic", "FAMILIAR_NO_ROBOT": "1"}
    subprocess.run(
        [sys.executable, "-m", "familiar_brain.cli", "index", "rules", str(p2014), "--edition", "2014", "--config", str(e2e_stack.cfg_path)],
        cwd=str(REPO_ROOT), env=env, check=True, timeout=15,
    )
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_tool_roundtrip("lookup_rule", {"query": "Old Exhaustion"}, "Six levels in 2014.")
    e2e_stack.client.post_turn("Rocky, how does old exhaustion work?")
    tool_out = [i["output"] for i in e2e_stack.llm.responses_requests()[-1]["body"]["input"] if i.get("type") == "function_call_output"][-1]
    assert "2014 rules" in tool_out


@pytest.mark.m1
def test_t1_f11_4_cli_index_rules_stores_edition_2024_in_sqlite(e2e_stack: StackHandle, tmp_path: Path):
    pdf = _make_sample_rules_pdf(tmp_path / "srd_v521.pdf", "Topple Mastery", "Topple weapon mastery forces a Constitution save.")
    env = {**os.environ, "FAMILIAR_EMBEDDER": "deterministic", "FAMILIAR_NO_ROBOT": "1"}
    proc = subprocess.run(
        [sys.executable, "-m", "familiar_brain.cli", "index", "rules", str(pdf), "--edition", "2024", "--config", str(e2e_stack.cfg_path)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    conn = sqlite3.connect(str(e2e_stack.db_path))
    rows = conn.execute("SELECT corpus, meta FROM chunks WHERE corpus='rules'").fetchall()
    conn.close()
    assert rows and any(json.loads(meta).get("edition") == "2024" for _, meta in rows)


@pytest.mark.m1
def test_t1_f11_5_repo_srd521_or_documented_2024_priority(e2e_stack: StackHandle, tmp_path: Path):
    srd521 = REPO_ROOT / "data" / "rules" / "SRD_CC_v5.2.1.pdf"
    if srd521.exists():
        conn = sqlite3.connect(str(REPO_ROOT / "data" / "familiar.db"))
        rows = conn.execute("SELECT COUNT(*) FROM chunks WHERE corpus='rules' AND meta LIKE '%2024%'").fetchone()
        conn.close()
        assert rows and rows[0] > 0
    else:
        policy = (REPO_ROOT / "docs" / "notes-policy.md").read_text(encoding="utf-8")
        assert "SRD_CC_v5.2.1.pdf" in policy and "--edition 2024" in policy


# ===========================================================================
# F12: Sep 11 Mock Session Purge (m1)
# ===========================================================================

@pytest.mark.m1
def test_t1_f12_1_repo_sessions_has_zero_sep11_logs():
    sessions = REPO_ROOT / "data" / "sessions"
    assert list(sessions.glob("20260911-*.jsonl")) == []


@pytest.mark.m1
def test_t1_f12_2_repo_sep16_session_preserved_and_valid_jsonl():
    sessions = REPO_ROOT / "data" / "sessions"
    assert list(sessions.glob("20260911-*")) == []
    sep16_jsonl = sessions / "20260916-182232-0189cc.jsonl"
    sep16_summary = sessions / "20260916-182232-0189cc.summary.md"
    assert sep16_jsonl.exists() or sep16_summary.exists()
    if sep16_jsonl.exists():
        lines = [ln for ln in sep16_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) > 0
        for ln in lines:
            obj = json.loads(ln)
            assert "text" in obj and "decision" in obj
    if sep16_summary.exists():
        txt = sep16_summary.read_text(encoding="utf-8")
        assert len(txt.strip()) > 0 and "## Scenes" in txt


@pytest.mark.m1
def test_t1_f12_3_api_sessions_lists_only_valid_sessions_no_sep11(e2e_stack: StackHandle):
    sessions = REPO_ROOT / "data" / "sessions"
    copied = 0
    for name in ("20260916-182232-0189cc.jsonl", "20260916-182232-0189cc.summary.md"):
        src = sessions / name
        if src.exists():
            shutil.copy2(src, e2e_stack.sessions_dir / src.name)
            copied += 1
    assert copied > 0, "Expected preserved Sep 16 session artifact in data/sessions/"
    r = e2e_stack.client.list_sessions()
    assert r.status_code == 200
    ids = [s["id"] for s in r.json()["sessions"]]
    assert "20260916-182232-0189cc" in ids
    assert not any(i.startswith("20260911-") for i in ids)


@pytest.mark.m1
def test_t1_f12_4_repo_db_contains_zero_sep11_chunks():
    db = REPO_ROOT / "data" / "familiar.db"
    assert db.exists()
    conn = sqlite3.connect(str(db))
    count = conn.execute("SELECT COUNT(*) FROM chunks WHERE source LIKE '20260911-%'").fetchone()[0]
    conn.close()
    assert count == 0


@pytest.mark.m1
def test_t1_f12_5_live_uninvited_rocky_fixture_evaluates_to_ignore(e2e_stack: StackHandle):
    fix = REPO_ROOT / "tests" / "fixtures" / "live_uninvited_rocky.jsonl"
    row = json.loads(fix.read_text(encoding="utf-8").splitlines()[0])
    e2e_stack.llm.queue_chat("IGNORE")
    r = e2e_stack.client.post_turn(row["text"])
    assert r.status_code == 200
    assert r.text == ""
    assert r.tool_names and set(r.tool_names) <= {"idle_do_nothing", "head_tracking", "play_emotion"}


# ===========================================================================
# F13: Live + Checkpoint Memory Updates (m2)
# ===========================================================================

@pytest.mark.m2
def test_t1_f13_1_live_indexer_indexes_session_facts_before_session_end(e2e_stack: StackHandle):
    e2e_stack.client.start_session()
    e2e_stack.client.note("The brass key opens the vault under the Jade Halls.")
    for i in range(8):
        e2e_stack.llm.queue_response_text(f"Noted {i}.")
        e2e_stack.client.post_turn(f"Rocky, remember clue number {i}: the silver lantern glows blue.")
    # Wait past LiveIndexer.TICK_S (5.0s) for background batch flush
    deadline = time.monotonic() + 8.0
    found = False
    while time.monotonic() < deadline:
        e2e_stack.llm.queue_tool_roundtrip("recall", {"query": "brass key vault Jade Halls", "scope": "sessions"}, "Yes, the brass key.")
        e2e_stack.client.post_turn("Rocky, recall the brass key?")
        outs = [i["output"] for i in e2e_stack.llm.responses_requests()[-1]["body"]["input"] if i.get("type") == "function_call_output"]
        if outs and "brass key" in outs[-1].lower():
            found = True
            break
        time.sleep(0.25)
    assert found, "LiveIndexer did not index active session notes/turns before session end"


@pytest.mark.m2
def test_t1_f13_2_checkpoint_force_summarises_and_updates_campaign_memory(e2e_stack: StackHandle):
    sid = e2e_stack.client.start_session().json()["session"]
    e2e_stack.llm.queue_response_text("Onward.")
    e2e_stack.client.post_turn("Rocky, we entered the sunken observatory in Old Town.")
    e2e_stack.client.note("Met Archivist Vane at the sunken observatory.")
    e2e_stack.llm.queue_chat("## Scenes\n- Entered the sunken observatory.\n## NPCs met\n- Archivist Vane.\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Campaign memory\n## Active threads\n- Observatory mystery with Archivist Vane.\n")
    cp = e2e_stack.client.checkpoint(force=True)
    assert cp.status_code == 200, cp.text
    data = cp.json()
    assert data["ok"] is True and data["session"] == sid and data["campaign_updated"] is True
    assert (e2e_stack.sessions_dir / "campaign-memory.md").exists()


@pytest.mark.m2
def test_t1_f13_3_checkpoint_keeps_session_active_and_appending(e2e_stack: StackHandle):
    sid = e2e_stack.client.start_session().json()["session"]
    e2e_stack.llm.queue_response_text("Before break.")
    e2e_stack.client.post_turn("Rocky, log phase one.")
    e2e_stack.llm.queue_chat("## Scenes\n- Phase one.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Campaign memory\n- Phase one.\n")
    assert e2e_stack.client.checkpoint(force=True).status_code == 200
    assert e2e_stack.client.get_state()["session"] == sid
    e2e_stack.llm.queue_response_text("After break.")
    e2e_stack.client.post_turn("Rocky, log phase two after break.")
    jsonl = (e2e_stack.sessions_dir / f"{sid}.jsonl").read_text(encoding="utf-8")
    assert "phase one" in jsonl and "phase two after break" in jsonl


@pytest.mark.m2
def test_t1_f13_4_session_end_after_checkpoint_writes_full_summary(e2e_stack: StackHandle):
    sid = e2e_stack.client.start_session().json()["session"]
    e2e_stack.llm.queue_response_text("Part A.")
    e2e_stack.client.post_turn("Rocky, we fought the clockwork hound.")
    e2e_stack.llm.queue_chat("## Scenes\n- Fought clockwork hound.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Campaign memory\n- Clockwork hound.\n")
    assert e2e_stack.client.checkpoint(force=True).status_code == 200

    e2e_stack.llm.queue_response_text("Part B.")
    e2e_stack.client.post_turn("Rocky, we recovered the silver astrolabe.")
    e2e_stack.llm.queue_chat("## Scenes\n- Fought clockwork hound and recovered silver astrolabe.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- Silver astrolabe.\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Campaign memory\n- Clockwork hound and silver astrolabe.\n")
    end_r = e2e_stack.client.end_session()
    assert end_r.status_code == 200
    summary_file = e2e_stack.sessions_dir / f"{sid}.summary.md"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not summary_file.exists():
        time.sleep(0.1)
    assert summary_file.exists()
    assert "astrolabe" in summary_file.read_text(encoding="utf-8").lower()


@pytest.mark.m2
def test_t1_f13_5_post_checkpoint_turns_retrieve_checkpointed_facts(e2e_stack: StackHandle):
    e2e_stack.client.start_session()
    e2e_stack.llm.queue_response_text("Noted.")
    e2e_stack.client.post_turn("Rocky, remember Captain Voss gave us the obsidian sigil.")
    e2e_stack.llm.queue_chat("## Scenes\n- Captain Voss gave the obsidian sigil.\n## NPCs met\n- Captain Voss.\n## Decisions\n- none\n## Loot\n- Obsidian sigil.\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Campaign memory\n- Captain Voss and the obsidian sigil.\n")
    assert e2e_stack.client.checkpoint(force=True).status_code == 200
    e2e_stack.llm.queue_tool_roundtrip("recall", {"query": "obsidian sigil Captain Voss", "scope": "all"}, "Captain Voss gave it.")
    e2e_stack.client.post_turn("Rocky, who gave us the obsidian sigil?")
    instr = e2e_stack.llm.last_instructions()
    assert "obsidian sigil" in instr.lower() or "voss" in instr.lower()


# ===========================================================================
# F14: Memory Safety Guarantees (m2)
# ===========================================================================

@pytest.mark.m2
def test_t1_f14_1_checkpoint_unforced_while_unmuted_refuses_409_with_zero_llm(e2e_stack: StackHandle):
    e2e_stack.client.start_session()
    e2e_stack.llm.reset_requests()
    r = e2e_stack.client.checkpoint()
    assert r.status_code == 409
    assert r.json().get("ok") is False
    assert e2e_stack.llm.responses_requests() == []


@pytest.mark.m2
def test_t1_f14_2_checkpoint_unforced_succeeds_when_muted_or_deep_night(e2e_stack: StackHandle):
    e2e_stack.client.start_session()
    e2e_stack.client.note("Quiet break note about the belltower.")
    e2e_stack.client.mute()
    e2e_stack.llm.queue_chat("## Scenes\n- Belltower note.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Campaign memory\n- Belltower.\n")
    r1 = e2e_stack.client.checkpoint(force=False)
    assert r1.status_code == 200 and r1.json()["ok"] is True

    e2e_stack.client.unmute()
    e2e_stack.client.deep_night(True)
    e2e_stack.llm.queue_chat("## Scenes\n- Belltower night.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Campaign memory\n- Belltower night.\n")
    r2 = e2e_stack.client.checkpoint(force=False)
    assert r2.status_code == 200 and r2.json()["ok"] is True


@pytest.mark.m2
def test_t1_f14_3_learn_while_session_active_returns_paused_zero_llm(e2e_stack: StackHandle):
    # Drop an unsummarised past session
    past = e2e_stack.sessions_dir / "20260901-000000-past01.jsonl"
    past.write_text('{"ts": 1.0, "text": "Rocky, hello", "decision": "ADDRESSED", "spoke": true}\n', encoding="utf-8")
    e2e_stack.client.start_session()
    e2e_stack.llm.reset_requests()
    r = e2e_stack.client.learn()
    assert r.status_code == 200
    assert r.json()["paused"] is True
    time.sleep(0.15)
    assert e2e_stack.llm.responses_requests() == []


@pytest.mark.m2
def test_t1_f14_4_purge_past_session_removes_files_and_rebuilds_campaign_memory(e2e_stack: StackHandle):
    sid = "20260902-120000-old001"
    (e2e_stack.sessions_dir / f"{sid}.jsonl").write_text('{"ts": 1.0, "text": "Rocky, secret", "decision": "ADDRESSED", "spoke": true}\n', encoding="utf-8")
    (e2e_stack.sessions_dir / f"{sid}.summary.md").write_text("## Scenes\n- Old purged tavern.\n", encoding="utf-8")
    (e2e_stack.sessions_dir / "campaign-memory.md").write_text("# Memory\n- Old purged tavern.\n", encoding="utf-8")
    r = e2e_stack.client.purge_session(sid)
    assert r.status_code == 200 and r.json()["purged"] is True
    assert not (e2e_stack.sessions_dir / f"{sid}.jsonl").exists()
    assert not (e2e_stack.sessions_dir / f"{sid}.summary.md").exists()


@pytest.mark.m2
def test_t1_f14_5_session_start_and_end_does_not_leak_live_index_threads(e2e_stack: StackHandle):
    import threading
    before = sum(1 for t in threading.enumerate() if "live-index" in t.name)
    for _ in range(3):
        e2e_stack.client.start_session()
        e2e_stack.llm.queue_response_text("Ack.")
        e2e_stack.client.post_turn("Rocky, ping.")
        e2e_stack.llm.queue_chat("## Scenes\n- Ping.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
        e2e_stack.llm.queue_chat("# Memory\n- Ping.\n")
        e2e_stack.client.end_session()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        after = sum(1 for t in threading.enumerate() if "live-index" in t.name)
        if after <= before:
            break
        time.sleep(0.05)
    assert sum(1 for t in threading.enumerate() if "live-index" in t.name) <= before


# ===========================================================================
# F15: Deep Night & /api/state Telemetry (m2)
# ===========================================================================

@pytest.mark.m2
def test_t1_f15_1_api_state_exposes_all_required_telemetry_keys(e2e_stack: StackHandle):
    st = e2e_stack.client.get_state()
    for key in (
        "muted", "asleep", "mode", "tracking", "logging", "session",
        "session_started_at", "last_spoke_at", "last_turn_ms",
        "recent", "transcript", "speakers", "robot",
    ):
        assert key in st, f"Missing {key!r} in GET /api/state"


@pytest.mark.m2
def test_t1_f15_2_deep_night_true_sleeps_without_muting_and_silences_turns(e2e_stack: StackHandle):
    dn = e2e_stack.client.deep_night(True)
    assert dn.status_code == 200
    assert dn.json()["asleep"] is True and dn.json()["muted"] is False
    st = e2e_stack.client.get_state()
    assert st["asleep"] is True and st["muted"] is False
    r = e2e_stack.client.post_turn("Rocky, wake up!")
    assert r.status_code == 200
    assert r.text == ""
    assert r.tool_names and set(r.tool_names) <= {"idle_do_nothing", "head_tracking", "play_emotion"}
    r2 = e2e_stack.client.post_turn("Rocky, still sleeping?")
    assert r2.status_code == 200 and r2.text == ""
    assert "idle_do_nothing" in r2.tool_names


@pytest.mark.m2
def test_t1_f15_3_deep_night_false_wakes_and_queues_awake_cue(e2e_stack: StackHandle):
    e2e_stack.client.deep_night(True)
    e2e_stack.client.post_turn("Rocky, quiet")  # flush asleep cue
    dn = e2e_stack.client.deep_night(False)
    assert dn.status_code == 200 and dn.json()["asleep"] is False
    e2e_stack.llm.queue_response_text("Good morning!")
    r = e2e_stack.client.post_turn("Rocky, good morning!")
    assert r.status_code == 200
    assert "head_tracking" in r.tool_names and "play_emotion" in r.tool_names
    emotions = [args.get("emotion") for name, _, args in r.tool_calls if name == "play_emotion"]
    assert "welcoming" in emotions


@pytest.mark.m2
def test_t1_f15_4_session_start_and_end_update_session_started_at(e2e_stack: StackHandle):
    assert e2e_stack.client.get_state()["session_started_at"] is None
    e2e_stack.client.start_session()
    st1 = e2e_stack.client.get_state()
    assert isinstance(st1["session_started_at"], (int, float))
    e2e_stack.client.end_session()
    st2 = e2e_stack.client.get_state()
    assert st2["session"] is None and st2["session_started_at"] is None


@pytest.mark.m2
def test_t1_f15_5_api_state_transcript_records_rolling_turns_in_order(e2e_stack: StackHandle):
    e2e_stack.write_config(follow_up_window_s=0.1)
    e2e_stack.restart_brain()
    e2e_stack.llm.queue_response_text("First reply.")
    e2e_stack.llm.queue_chat("IGNORE")
    e2e_stack.client.post_turn("Rocky, first turn")
    time.sleep(0.25)
    e2e_stack.client.post_turn("Ordinary table chatter")
    tr = e2e_stack.client.get_state()["transcript"]
    assert len(tr) >= 2
    assert tr[-2]["decision"] == "ADDRESSED" and tr[-2]["spoke"] is True
    assert tr[-1]["decision"] == "IGNORE" and tr[-1]["spoke"] is False
    for row in tr[-2:]:
        assert {"text", "speaker", "decision", "spoke", "ts"} <= set(row.keys())


# ===========================================================================
# F16: Transparent CPU-Only Audio Tap (m3)
# ===========================================================================

@pytest.mark.m3
def test_t1_f16_1_audio_tap_proxies_pool_and_usage_http(e2e_stack: StackHandle):
    tap = e2e_stack.start_audio_tap()
    p = requests.get(f"{tap.base_url}/v1/pool", timeout=3.0)
    u = requests.get(f"{tap.base_url}/v1/usage", timeout=3.0)
    assert p.status_code == 200 and p.json().get("ready") is True
    assert u.status_code == 200 and "connections" in u.json()


@pytest.mark.m3
def test_t1_f16_2_audio_tap_forwards_client_ws_frames_identically(e2e_stack: StackHandle):
    tap = e2e_stack.start_audio_tap()
    frame1 = {"type": "session.update", "session": {"modalities": ["audio", "text"]}}
    frame2 = {"type": "input_audio_buffer.append", "audio": pcm16_to_b64(b"\x01\x00" * 160)}
    with tap.ws_client() as ws:
        ws.send_json(frame1)
        ws.send_json(frame2)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(e2e_stack.s2s.ws_messages) < 2:
            time.sleep(0.05)
    assert frame1 in e2e_stack.s2s.ws_messages
    assert frame2 in e2e_stack.s2s.ws_messages


@pytest.mark.m3
def test_t1_f16_3_audio_tap_forwards_server_to_client_ws_events(e2e_stack: StackHandle):
    tap = e2e_stack.start_audio_tap()
    with tap.ws_client() as ws:
        ws.send_json({"type": "session.update"})
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not e2e_stack.s2s.ws_messages:
            time.sleep(0.05)
        e2e_stack.s2s.broadcast_ws({"type": "response.text.delta", "delta": "Hello from S2S"})
        ev = ws.recv_json(timeout=2.0)
        assert ev == {"type": "response.text.delta", "delta": "Hello from S2S"}


@pytest.mark.m3
def test_t1_f16_4_audio_tap_identifies_brokk_vs_till_on_cpu(e2e_stack: StackHandle, audio_samples):
    assert e2e_stack.client.enroll_speaker({"role": "Brokk", "audio_b64": audio_samples["brokk_b64"], "sample_rate": 16000}).status_code == 200
    assert e2e_stack.client.enroll_speaker({"role": "Till", "audio_b64": audio_samples["till_b64"], "sample_rate": 16000}).status_code == 200
    tap = e2e_stack.start_audio_tap()
    with tap.ws_client() as ws:
        ws.send_audio_pcm16(audio_samples["brokk_pcm_2"])
        ws.send_json({"type": "input_audio_buffer.commit"})
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if tap.current().get("role") == "Brokk":
                break
            time.sleep(0.05)
        assert tap.current().get("role") == "Brokk"

        ws.send_audio_pcm16(audio_samples["till_pcm"])
        ws.send_json({"type": "input_audio_buffer.commit"})
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if tap.current().get("role") == "Till":
                break
            time.sleep(0.05)
        assert tap.current().get("role") == "Till"


@pytest.mark.m3
def test_t1_f16_5_audio_tap_forwarding_latency_overhead_under_10ms(e2e_stack: StackHandle, audio_samples):
    e2e_stack.client.enroll_speaker({"role": "DM", "audio_b64": audio_samples["dm_b64"], "sample_rate": 16000})
    tap = e2e_stack.start_audio_tap()
    payload = {"type": "input_audio_buffer.append", "audio": pcm16_to_b64(audio_samples["dm_pcm"][:3200])}
    with tap.ws_client() as ws:
        t0 = time.monotonic()
        for _ in range(20):
            ws.send_json(payload)
        pong = ws.ping(b"sync")
        assert pong == b"sync"
        avg_ms = ((time.monotonic() - t0) * 1000.0) / 20.0
    assert avg_ms < 10.0, f"Per-frame forwarding overhead too high: {avg_ms:.2f} ms"


# ===========================================================================
# F17: Opt-In Speaker Enrollment APIs (m3)
# ===========================================================================

@pytest.mark.m3
def test_t1_f17_1_get_speakers_initial_empty_shape(e2e_stack: StackHandle):
    r = e2e_stack.client.get_speakers()
    assert r.status_code == 200
    d = r.json()
    assert d["enrolled"] == []
    assert d["last_speaker"] is None
    assert d["capturing"] is False
    assert isinstance(d["enabled"], bool)


@pytest.mark.m3
def test_t1_f17_2_enroll_speaker_with_samples_persists(e2e_stack: StackHandle, audio_samples):
    r = e2e_stack.client.enroll_speaker({"role": "DM", "samples": audio_samples["dm_samples"], "sample_rate": 16000})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True and d["role"] == "DM" and "DM" in d["enrolled"]
    assert "DM" in e2e_stack.client.get_speakers().json()["enrolled"]


@pytest.mark.m3
def test_t1_f17_3_enroll_speaker_with_audio_b64_alongside_dm(e2e_stack: StackHandle, audio_samples):
    assert e2e_stack.client.enroll_speaker({"role": "DM", "samples": audio_samples["dm_samples"], "sample_rate": 16000}).status_code == 200
    r = e2e_stack.client.enroll_speaker({"role": "Brokk", "audio_b64": audio_samples["brokk_b64"], "sample_rate": 16000})
    assert r.status_code == 200
    assert set(r.json()["enrolled"]) >= {"DM", "Brokk"}


@pytest.mark.m3
def test_t1_f17_4_enroll_speaker_live_capture_from_audio_tap(e2e_stack: StackHandle, audio_samples):
    tap = e2e_stack.start_audio_tap()
    result_box: list[requests.Response] = []

    def _enroll():
        result_box.append(e2e_stack.client.enroll_speaker({"role": "Krag", "duration_s": 0.6}))

    import threading
    th = threading.Thread(target=_enroll, daemon=True)
    th.start()
    time.sleep(0.1)
    with tap.ws_client() as ws:
        ws.send_audio_pcm16(audio_samples["krag_pcm"])
        time.sleep(0.7)
    th.join(timeout=5.0)
    assert result_box and result_box[0].status_code == 200, result_box[0].text if result_box else "no response"
    assert "Krag" in result_box[0].json()["enrolled"]


@pytest.mark.m3
def test_t1_f17_5_delete_single_speaker_and_clear_all(e2e_stack: StackHandle, audio_samples):
    e2e_stack.client.enroll_speaker({"role": "DM", "audio_b64": audio_samples["dm_b64"], "sample_rate": 16000})
    e2e_stack.client.enroll_speaker({"role": "Krag", "audio_b64": audio_samples["krag_b64"], "sample_rate": 16000})
    d1 = e2e_stack.client.delete_speaker("Krag")
    assert d1.status_code == 200 and "Krag" not in d1.json()["enrolled"] and "DM" in d1.json()["enrolled"]
    d2 = e2e_stack.client.clear_speakers()
    assert d2.status_code == 200 and d2.json()["enrolled"] == []


# ===========================================================================
# F18: Speaker-Aware Gating & Persona (m3)
# ===========================================================================

@pytest.mark.m3
def test_t1_f18_1_brokk_speaker_injects_master_context_and_keeps_secrets(e2e_stack: StackHandle):
    e2e_stack.llm.queue_response_text("Yes, master Brokk.")
    r = e2e_stack.client.post_turn(
        "Rocky, what's the plan?",
        headers={"X-Familiar-Speaker-Role": "Brokk", "X-Familiar-Speaker-Confidence": "0.91"},
    )
    assert r.status_code == 200
    instr = e2e_stack.llm.last_instructions()
    assert "Brokk" in instr and "master" in instr.lower()
    assert "Vacoran Amulet" in instr


@pytest.mark.m3
def test_t1_f18_2_dm_speaker_never_interrupted_inside_followup_window(e2e_stack: StackHandle):
    e2e_stack.llm.queue_response_text("Ready, Brokk.")
    r1 = e2e_stack.client.post_turn("Rocky, are you ready?", headers={"X-Familiar-Speaker-Role": "Brokk"})
    assert r1.text
    # Inside 8s follow-up window, DM narrates without naming Rocky -> must IGNORE!
    r2 = e2e_stack.client.post_turn(
        "You step onto the cobblestones of Northreach Crossing as fog curls around your boots.",
        headers={"X-Familiar-Speaker-Role": "DM"},
    )
    assert r2.status_code == 200
    assert r2.text == ""
    assert "idle_do_nothing" in r2.tool_names


@pytest.mark.m3
def test_t1_f18_3_dm_voice_triggers_deep_night_sleep_and_wake(e2e_stack: StackHandle):
    r1 = e2e_stack.client.post_turn("Deep night falls over Nephria.", headers={"X-Familiar-Speaker-Role": "DM"})
    assert r1.status_code == 200 and e2e_stack.client.get_state()["asleep"] is True
    e2e_stack.llm.queue_response_text("Back awake.")
    r2 = e2e_stack.client.post_turn("Deep night is over, Rocky, good morning.", headers={"X-Familiar-Speaker-Role": "DM"})
    assert r2.status_code == 200 and e2e_stack.client.get_state()["asleep"] is False


@pytest.mark.m3
def test_t1_f18_4_krag_speaker_logs_role_in_jsonl_and_state_transcript(e2e_stack: StackHandle):
    e2e_stack.client.start_session()
    e2e_stack.llm.queue_response_text("Right behind you, Krag.")
    r = e2e_stack.client.post_turn(
        "Rocky, watch the corridor!",
        headers={"X-Familiar-Speaker-Role": "Krag", "X-Familiar-Speaker-Confidence": "0.88"},
    )
    assert r.status_code == 200
    instr = e2e_stack.llm.last_instructions()
    assert "Krag" in instr
    tr = e2e_stack.client.get_state()["transcript"]
    assert tr and tr[-1]["speaker"] == "Krag"
    path = Path(e2e_stack.client.end_session().json()["path"])
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert rows and rows[-1].get("speaker") == "Krag"


@pytest.mark.m3
def test_t1_f18_5_omitted_speaker_header_with_zero_enrolled_matches_baseline(e2e_stack: StackHandle):
    e2e_stack.llm.queue_response_text("At your side.")
    e2e_stack.llm.queue_response_text("Follow-up answer.")
    r1 = e2e_stack.client.post_turn("Rocky, hello.")
    r2 = e2e_stack.client.post_turn("And what else?")
    assert r1.text and r2.text
    assert e2e_stack.client.get_state()["transcript"][-1]["speaker"] is None


# ===========================================================================
# F19: Zero-Build Companion UI (m4)
# ===========================================================================

@pytest.mark.m4
def test_t1_f19_1_get_root_serves_art_deco_noir_html(e2e_stack: StackHandle):
    r = e2e_stack.client.get_ui("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("Content-Type", "")
    html = r.text
    assert "viewport" in html and "Rocky" in html


@pytest.mark.m4
def test_t1_f19_2_get_static_styles_css_has_art_deco_and_touch_targets(e2e_stack: StackHandle):
    r = e2e_stack.client.get_ui("/static/styles.css")
    assert r.status_code == 200
    assert "text/css" in r.headers.get("Content-Type", "")
    css = r.text
    assert "--" in css and "48px" in css


@pytest.mark.m4
def test_t1_f19_3_get_static_app_js_polls_api_state(e2e_stack: StackHandle):
    r = e2e_stack.client.get_ui("/static/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers.get("Content-Type", "")
    assert "/api/state" in r.text


@pytest.mark.m4
def test_t1_f19_4_static_assets_have_zero_external_cdn_urls(e2e_stack: StackHandle):
    for path in ("/", "/static/styles.css", "/static/app.js"):
        txt = e2e_stack.client.get_ui(path).text
        assert not re.search(r"https?://|//cdn|fonts\.googleapis\.com", txt, re.I), f"External URL in {path}"


@pytest.mark.m4
def test_t1_f19_5_gitignore_does_not_ignore_familiar_brain_static():
    gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in gi.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert "familiar_brain/static/" not in lines and "familiar_brain/static" not in lines


# ===========================================================================
# F20: Companion UI Controls (m4)
# ===========================================================================

@pytest.mark.m4
def test_t1_f20_1_html_exposes_all_14_companion_controls(e2e_stack: StackHandle):
    html = e2e_stack.client.get_ui("/").text.lower()
    for token in (
        "mute", "unmute", "speak", "mode", "deep", "note",
        "checkpoint", "start", "end", "purge", "reindex", "reconnect", "enroll",
    ):
        assert token in html, f"Missing UI control token {token!r}"


@pytest.mark.m4
def test_t1_f20_2_app_js_wires_all_backend_api_endpoints(e2e_stack: StackHandle):
    js = e2e_stack.client.get_ui("/static/app.js").text
    for endpoint in (
        "/api/mute", "/api/unmute", "/api/mute_and_stop_logging", "/api/speak_now",
        "/api/mode", "/api/deep_night", "/api/note", "/api/session/checkpoint",
        "/api/session/start", "/api/session/end", "/api/session/",
        "/api/reindex", "/api/robot/reconnect", "/api/speakers",
    ):
        assert endpoint in js, f"Missing endpoint wiring {endpoint!r} in app.js"


@pytest.mark.m4
def test_t1_f20_3_full_ui_control_api_sequence_updates_state(e2e_stack: StackHandle):
    sid = e2e_stack.client.start_session().json()["session"]
    assert e2e_stack.client.mute().status_code == 200
    assert e2e_stack.client.get_state()["muted"] is True
    assert e2e_stack.client.unmute().status_code == 200
    assert e2e_stack.client.get_state()["muted"] is False
    assert e2e_stack.client.set_mode("thinking").status_code == 200
    assert e2e_stack.client.get_state()["mode"] == "thinking"
    assert e2e_stack.client.deep_night(True).status_code == 200
    assert e2e_stack.client.get_state()["asleep"] is True
    assert e2e_stack.client.deep_night(False).status_code == 200
    assert e2e_stack.client.note("UI note saved.").status_code == 200
    assert e2e_stack.client.reindex().status_code == 200
    e2e_stack.llm.queue_chat("## Scenes\n- UI note.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Memory\n- UI note.\n")
    assert e2e_stack.client.checkpoint(force=True).status_code == 200
    assert e2e_stack.client.end_session().status_code == 200
    assert e2e_stack.client.purge_session(sid).status_code == 200


@pytest.mark.m4
def test_t1_f20_4_ui_speaker_enroll_list_delete_syncs_with_api_state(e2e_stack: StackHandle, audio_samples):
    assert e2e_stack.client.enroll_speaker({"role": "DM", "audio_b64": audio_samples["dm_b64"], "sample_rate": 16000}).status_code == 200
    assert e2e_stack.client.enroll_speaker({"role": "Brokk", "audio_b64": audio_samples["brokk_b64"], "sample_rate": 16000}).status_code == 200
    assert set(e2e_stack.client.get_state()["speakers"]["enrolled"]) >= {"DM", "Brokk"}
    assert e2e_stack.client.delete_speaker("Brokk").status_code == 200
    assert "Brokk" not in e2e_stack.client.get_state()["speakers"]["enrolled"]
    assert e2e_stack.client.clear_speakers().status_code == 200
    assert e2e_stack.client.get_state()["speakers"]["enrolled"] == []


@pytest.mark.m4
def test_t1_f20_5_ui_checkpoint_updates_memory_without_stopping_timer(e2e_stack: StackHandle):
    e2e_stack.client.start_session()
    t_start = e2e_stack.client.get_state()["session_started_at"]
    e2e_stack.client.note("Discovered a hidden stairwell behind the tapestry.")
    e2e_stack.llm.queue_chat("## Scenes\n- Hidden stairwell.\n## NPCs met\n- none\n## Decisions\n- none\n## Loot\n- none\n## Open threads\n- none\n## Memorable lines\n- none\n")
    e2e_stack.llm.queue_chat("# Memory\n- Hidden stairwell.\n")
    cp = e2e_stack.client.checkpoint(force=True)
    assert cp.status_code == 200 and cp.json()["campaign_updated"] is True
    assert e2e_stack.client.get_state()["session_started_at"] == t_start


# ===========================================================================
# F21: Live Status Strip & Robot API (m4)
# ===========================================================================

@pytest.mark.m4
def test_t1_f21_1_robot_status_returns_safe_json_without_touching_physical_robot(e2e_stack: StackHandle):
    r = e2e_stack.client.robot_status()
    assert r.status_code == 200
    d = r.json()
    assert {"backend_connected", "host", "last_checked_at", "detail", "disabled"} <= set(d.keys())
    assert d["disabled"] is True and d["backend_connected"] is False


@pytest.mark.m4
def test_t1_f21_2_robot_reconnect_against_fake_reachy_succeeds(e2e_stack: StackHandle):
    r = e2e_stack.client.robot_reconnect({
        "host": "127.0.0.1",
        "daemon_port": e2e_stack.reachy.daemon_port,
        "rpc_port": e2e_stack.reachy.rpc_port,
        "host_ip": "127.0.0.1",
        "port": e2e_stack.s2s.port,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is True and d["backend_connected"] is True


@pytest.mark.m4
def test_t1_f21_3_api_state_reflects_robot_connected_after_reconnect(e2e_stack: StackHandle):
    e2e_stack.client.robot_reconnect({
        "host": "127.0.0.1",
        "daemon_port": e2e_stack.reachy.daemon_port,
        "rpc_port": e2e_stack.reachy.rpc_port,
        "host_ip": "127.0.0.1",
        "port": e2e_stack.s2s.port,
    })
    st = e2e_stack.client.get_state()
    assert st["robot"]["backend_connected"] is True


@pytest.mark.m4
def test_t1_f21_4_ui_html_includes_status_strip_and_transcript_elements(e2e_stack: StackHandle):
    html = e2e_stack.client.get_ui("/").text.lower()
    assert "transcript" in html and "latency" in html and "robot" in html


@pytest.mark.m4
def test_t1_f21_5_rolling_transcript_retains_speaker_tags_decisions_and_spoke(e2e_stack: StackHandle):
    e2e_stack.llm.queue_response_text("Yes, Brokk.")
    e2e_stack.client.post_turn("Rocky, ready?", headers={"X-Familiar-Speaker-Role": "Brokk"})
    e2e_stack.client.post_turn("The alley is quiet.", headers={"X-Familiar-Speaker-Role": "DM"})
    e2e_stack.llm.queue_chat("IGNORE")
    e2e_stack.client.post_turn("Pass the dice tray.")
    tr = e2e_stack.client.get_state()["transcript"]
    assert len(tr) >= 3
    assert [row["speaker"] for row in tr[-3:]] == ["Brokk", "DM", None]
    assert [row["decision"] for row in tr[-3:]] == ["ADDRESSED", "IGNORE", "IGNORE"]
    assert [row["spoke"] for row in tr[-3:]] == [True, False, False]
