"""start_all.sh keeps every launch's logs in its own run dir under data/logs/.

These tests run the real script under bash with stub binaries, so they run in WSL or Linux
only. On Windows a subprocess "bash" can resolve to WSL's bash.exe, which would see none of
the stubs. The script is copied into a tmp_path repo so nothing touches the worktree's data/,
and curl is stubbed to fail so no readiness probe reaches a real llama, brain or s2s port.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="runs start_all.sh under bash with stub binaries; run it in WSL (bash here may be WSL's bash.exe)",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _stub(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def stack(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "start_all.sh", repo / "scripts" / "start_all.sh")
    bin_dir = tmp_path / "bin"
    _stub(bin_dir / "curl", "exit 7")
    _stub(tmp_path / "llama" / "llama-server", 'echo "llama stub"')
    # The script stops at the brain's readiness wait once the brain exits, and its cleanup
    # then kills the other children; the pause lets llama and s2s write their line first.
    _stub(tmp_path / "brainenv" / "bin" / "familiar-brain", 'echo "brain stub"; sleep 0.5')
    _stub(tmp_path / "s2s" / "bin" / "python", 'echo "s2s stub"')
    env = {k: v for k, v in os.environ.items() if k != "RUN_DIR"}
    env.update(
        HOME=str(tmp_path / "home"),
        PATH=f"{bin_dir}:{env.get('PATH', '/usr/bin:/bin')}",
        LLAMA_BIN=str(tmp_path / "llama" / "llama-server"),
        S2S_VENV=str(tmp_path / "s2s"),
        UV_PROJECT_ENVIRONMENT=str(tmp_path / "brainenv"),
        STARTUP_TIMEOUT_S="3",
        HF_HUB_OFFLINE="1",
    )
    return {"repo": repo, "bin": bin_dir, "env": env}


def _run(stack: dict, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(stack["env"], **(extra_env or {}))
    return subprocess.run(
        ["bash", "scripts/start_all.sh", "--skip-robot", "--no-audio-tap"],
        cwd=stack["repo"],
        env=env,
        capture_output=True,
        text=True,
        timeout=8,
    )


def _run_dirs(repo: Path) -> list[Path]:
    logs = repo / "data" / "logs"
    return sorted(p for p in logs.iterdir() if p.is_dir() and not p.is_symlink())


def _wait_for_text(path: Path, timeout: float = 3.0) -> str:
    deadline = time.monotonic() + timeout
    while True:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if text.strip() or time.monotonic() > deadline:
            return text
        time.sleep(0.05)


def test_start_all_keeps_previous_run_logs(stack: dict):
    first = _run(stack)
    time.sleep(1.1)  # run dirs are named to the second
    second = _run(stack)

    runs = _run_dirs(stack["repo"])
    assert len(runs) == 2, (runs, first.stderr, second.stderr)
    for run in runs:
        assert "brain stub" in (run / "brain.log").read_text(encoding="utf-8")
        assert "llama stub" in _wait_for_text(run / "llama.log")
    data = stack["repo"] / "data"
    latest = data / "logs" / "latest"
    assert latest.is_symlink() and latest.resolve() == runs[-1].resolve()
    # Hard links, not symlinks: Windows cannot open WSL symlinks on the /mnt/c checkout.
    for name in ("llama", "brain", "s2s", "audio_tap", "robot_connect"):
        legacy = data / f"{name}.log"
        assert not legacy.is_symlink(), name
        assert os.path.samefile(legacy, runs[-1] / f"{name}.log"), name


def test_start_all_prunes_all_but_the_newest_20_run_dirs(stack: dict):
    logs = stack["repo"] / "data" / "logs"
    for i in range(25):
        (logs / f"20000101-0000{i:02d}").mkdir(parents=True)

    result = _run(stack)

    runs = _run_dirs(stack["repo"])
    assert len(runs) == 20, (runs, result.stderr)
    assert not (logs / "20000101-000000").exists()
    assert "brain stub" in (runs[-1] / "brain.log").read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform != "win32" and os.geteuid() == 0, reason="root can delete a read-only dir")
def test_start_all_still_launches_when_an_old_run_dir_cannot_be_pruned(stack: dict):
    logs = stack["repo"] / "data" / "logs"
    for i in range(21):
        (logs / f"20000101-0000{i:02d}").mkdir(parents=True)
    stuck = logs / "20000101-000000"
    (stuck / "brain.log").write_text("old", encoding="utf-8")
    stuck.chmod(0o555)
    try:
        result = _run(stack)
    finally:
        stuck.chmod(0o755)

    assert "could not prune" in result.stderr, result.stderr
    assert "brain stub" in (_run_dirs(stack["repo"])[-1] / "brain.log").read_text(encoding="utf-8")


def test_start_all_uses_the_run_dir_it_is_given(stack: dict):
    result = _run(stack, {"RUN_DIR": "data/logs/29991231-235959"})

    run = stack["repo"] / "data" / "logs" / "29991231-235959"
    assert "brain stub" in (run / "brain.log").read_text(encoding="utf-8"), result.stderr
    assert len(_run_dirs(stack["repo"])) == 1


def test_start_all_timestamps_child_lines_when_ts_is_installed(stack: dict):
    _stub(stack["bin"] / "ts", 'while IFS= read -r line; do echo "TS $line"; done')

    result = _run(stack)

    run = _run_dirs(stack["repo"])[-1]
    assert _wait_for_text(run / "brain.log").startswith("TS brain stub"), result.stderr
    # s2s stamps its own lines, so it is not piped through ts
    assert (run / "s2s.log").read_text(encoding="utf-8").startswith("s2s stub")
