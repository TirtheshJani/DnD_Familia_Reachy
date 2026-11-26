"""The speech-to-speech arguments start_all.sh launches with.

S2S_ARGS is evaluated by bash itself, so quoting and line continuations count as they do
at launch. Only the array assignment runs, never the rest of the script. On Windows a
subprocess "bash" can resolve to WSL's bash.exe, so this runs in WSL or Linux only.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="evaluates S2S_ARGS from start_all.sh under bash; run it in WSL (bash here may be WSL's bash.exe)",
)

START_ALL = Path(__file__).resolve().parent.parent / "scripts" / "start_all.sh"


def _s2s_args() -> list[str]:
    text = START_ALL.read_text(encoding="utf-8")
    block = re.search(r"^S2S_ARGS=\(\n.*?^\)$", text, flags=re.MULTILINE | re.DOTALL)
    assert block, "S2S_ARGS=( ... ) not found in start_all.sh"
    script = 'S2S_HOST=127.0.0.1\nS2S_PORT=1\n' + block.group(0) + '\nprintf "%s\\n" "${S2S_ARGS[@]}"\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10, check=True)
    return out.stdout.splitlines()


def test_start_all_does_not_set_finite_max_speech_ms() -> None:
    """s2s discards a segment longer than max_speech_ms with no transcript, so a player
    who talks past it is never heard. Left unset it is infinite."""
    args = _s2s_args()
    assert "--llm_backend" in args, f"S2S_ARGS did not evaluate as expected: {args}"
    assert "--max_speech_ms" not in args
