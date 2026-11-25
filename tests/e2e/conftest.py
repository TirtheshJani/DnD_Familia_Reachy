"""Pytest configuration, progressive markers, and strictly opaque-box E2E fixtures.

STRICT OPAQUE-BOX RULE:
- Register markers in ``pytest_configure`` without modifying ``pyproject.toml``.
- Never import private/internal symbols from ``familiar_brain`` (only public ``create_app``).
- Never connect to physical Reachy Mini (``10.0.0.100`` / ``reachy-mini.local``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import requests
from werkzeug.serving import make_server

from familiar_brain.app import create_app
from tests.e2e.fakes import (
    FakeReachyServer,
    FakeS2SServer,
    FakeUpstreamLLMServer,
    WsRealtimeClient,
    brokk_pcm,
    dm_pcm,
    find_free_port,
    krag_pcm,
    pcm16_to_b64,
    pcm16_to_float_samples,
    silence_pcm,
    stranger_pcm,
    till_pcm,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_ROBOT_HOSTS = frozenset({"10.0.0.100", "reachy-mini.local"})

DEFAULT_PERSONA_YAML = """\
name: Rocky
aliases: [rocky, rocki, rockie]
tts_speaker: Aiden
manner: |
  Attentive and dry, with a quick spark of delight or dismay when the dice turn.
  Curious about mortals and their habits. Answers in one or two sentences.
gestures:
  attentive: play_emotion:attentive
  amused: play_emotion:happy
  alarmed: play_emotion:surprised
  thoughtful: play_emotion:thinking
  proud: play_emotion:success
  sad: play_emotion:sad
  scared: play_emotion:scared
  asleep: play_emotion:sleepy
  awake: play_emotion:welcoming
  cheer: play_emotion:success
"""

DEFAULT_ROBOT_TOOLS = [
    {"type": "function", "name": "move_head", "parameters": {"type": "object", "properties": {"direction": {"type": "string"}}}},
    {"type": "function", "name": "play_emotion", "parameters": {"type": "object", "properties": {"emotion": {"type": "string"}}}},
    {"type": "function", "name": "stop_emotion", "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "idle_do_nothing", "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "head_tracking", "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}}}},
    {"type": "function", "name": "camera", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}}},
]


def pytest_configure(config: pytest.Config) -> None:
    markers = [
        "e2e: opaque-box end-to-end test suite",
        "m1: Milestone 1 features (F8-F12: campaign notes, lexicon, persona, 2024 rules, Sep 11 purge)",
        "m2: Milestone 2 features (F7, F13-F15: stall fix, live/checkpoint memory, safety, Deep Night)",
        "m3: Milestone 3 features (F16-F18: CPU audio tap, speaker enrollment, speaker-aware gating)",
        "m4: Milestone 4 features (F19-F21: zero-build Companion UI, controls, status strip, robot API)",
        "m5: Milestone 5 features (F1-F6: start_all.sh, robot_connect.py, launch.ps1, STT, Gemma 4, TTS)",
        "t1: Tier 1 happy-path E2E tests",
        "t2: Tier 2 edge, boundary, and error E2E tests",
        "t3: Tier 3 cross-feature pair E2E tests",
        "t4: Tier 4 real-world D&D table scenario E2E tests",
    ]
    for m in markers:
        config.addinivalue_line("markers", m)


def parse_sse_events(raw_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


@dataclass
class TurnResult:
    status_code: int
    content_type: str
    raw_text: str
    events: list[dict[str, Any]] = field(default_factory=list)
    json_body: dict[str, Any] | None = None

    @property
    def event_types(self) -> list[str]:
        return [str(ev.get("type", "")) for ev in self.events]

    @property
    def tool_calls(self) -> list[tuple[str, str, dict[str, Any]]]:
        out: list[tuple[str, str, dict[str, Any]]] = []
        for ev in self.events:
            if ev.get("type") == "response.output_item.done":
                item = ev.get("item") or {}
                if item.get("type") == "function_call":
                    raw_args = item.get("arguments") or "{}"
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed_args = {"_raw": raw_args}
                    out.append((str(item.get("name", "")), str(item.get("call_id", "")), parsed_args))
        return out

    @property
    def tool_names(self) -> list[str]:
        return [name for name, _, _ in self.tool_calls]

    @property
    def text(self) -> str:
        return "".join(
            str(ev.get("delta", ""))
            for ev in self.events
            if ev.get("type") == "response.output_text.delta"
        )

    @property
    def completed_count(self) -> int:
        return sum(1 for ev in self.events if ev.get("type") == "response.completed")


class HeadlessClient:
    """Opaque-box HTTP/SSE client driving familiar-brain over real loopback TCP."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def post_responses(
        self,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 15.0,
    ) -> requests.Response:
        return self.session.post(
            f"{self.base_url}/v1/responses",
            json=payload,
            headers=headers,
            timeout=timeout,
        )

    def post_turn(
        self,
        text: str,
        *,
        stream: bool = True,
        history: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        model: str = "familiar",
        image_b64: str | None = None,
        timeout: float = 15.0,
    ) -> TurnResult:
        if image_b64 is not None and not text:
            content: Any = [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"}]
        elif image_b64 is not None:
            content = [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            ]
        else:
            content = [{"type": "input_text", "text": text}]
        items = list(history or []) + [{"type": "message", "role": "user", "content": content}]
        payload: dict[str, Any] = {
            "model": model,
            "input": items,
            "tools": DEFAULT_ROBOT_TOOLS if tools is None else tools,
        }
        if stream:
            payload["stream"] = True
        resp = self.post_responses(payload, headers=headers, timeout=timeout)
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.text
        if "text/event-stream" in ctype:
            return TurnResult(
                status_code=resp.status_code,
                content_type=ctype,
                raw_text=raw,
                events=parse_sse_events(raw),
            )
        try:
            body = resp.json()
        except Exception:
            body = None
        return TurnResult(
            status_code=resp.status_code,
            content_type=ctype,
            raw_text=raw,
            json_body=body,
        )

    def get_state(self) -> dict[str, Any]:
        r = self.session.get(f"{self.base_url}/api/state", timeout=5.0)
        r.raise_for_status()
        return r.json()

    def mute(self) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/mute", json={}, timeout=5.0)

    def unmute(self) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/unmute", json={}, timeout=5.0)

    def mute_and_stop_logging(self) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/mute_and_stop_logging", json={}, timeout=5.0)

    def set_mode(self, mode: str) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/mode", json={"mode": mode}, timeout=5.0)

    def deep_night(self, asleep: bool | Any = ...) -> requests.Response:
        payload = {} if asleep is ... else {"asleep": asleep}
        return self.session.post(f"{self.base_url}/api/deep_night", json=payload, timeout=5.0)

    def speak_now(self) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/speak_now", json={}, timeout=5.0)

    def note(self, text: str) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/note", json={"text": text}, timeout=5.0)

    def reindex(self) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/reindex", json={}, timeout=10.0)

    def start_session(self) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/session/start", json={}, timeout=10.0)

    def end_session(self) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/session/end", json={}, timeout=10.0)

    def checkpoint(self, force: bool | None = None) -> requests.Response:
        body = {} if force is None else {"force": force}
        return self.session.post(f"{self.base_url}/api/session/checkpoint", json=body, timeout=10.0)

    def learn(self) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/learn", json={}, timeout=10.0)

    def list_sessions(self) -> requests.Response:
        return self.session.get(f"{self.base_url}/api/sessions", timeout=5.0)

    def purge_session(self, sid: str) -> requests.Response:
        return self.session.delete(f"{self.base_url}/api/session/{sid}", timeout=10.0)

    def get_speakers(self) -> requests.Response:
        return self.session.get(f"{self.base_url}/api/speakers", timeout=5.0)

    def enroll_speaker(self, payload: dict[str, Any]) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/speakers/enroll", json=payload, timeout=10.0)

    def delete_speaker(self, role: str) -> requests.Response:
        return self.session.delete(f"{self.base_url}/api/speakers/{role}", timeout=5.0)

    def clear_speakers(self) -> requests.Response:
        return self.session.delete(f"{self.base_url}/api/speakers", timeout=5.0)

    def robot_status(self) -> requests.Response:
        return self.session.get(f"{self.base_url}/api/robot/status", timeout=5.0)

    def robot_reconnect(self, payload: dict[str, Any] | None = None) -> requests.Response:
        return self.session.post(f"{self.base_url}/api/robot/reconnect", json=payload or {}, timeout=10.0)

    def get_ui(self, path: str = "/") -> requests.Response:
        return self.session.get(f"{self.base_url}{path}", timeout=5.0)


class AudioTapProcess:
    """Manages an opaque-box ``python -m familiar_brain.audio_tap`` subprocess."""

    def __init__(
        self,
        *,
        listen_port: int,
        upstream_port: int,
        control_port: int,
        cfg_path: Path,
        env: dict[str, str],
    ) -> None:
        mod_path = REPO_ROOT / "familiar_brain" / "audio_tap.py"
        if not mod_path.exists():
            raise RuntimeError("familiar_brain/audio_tap.py does not exist yet (requires M3)")
        self.listen_port = listen_port
        self.upstream_port = upstream_port
        self.control_port = control_port
        self.base_url = f"http://127.0.0.1:{listen_port}"
        self.ws_url = f"ws://127.0.0.1:{listen_port}/v1/realtime"
        self.control_url = f"http://127.0.0.1:{control_port}"
        cmd = [
            sys.executable,
            "-m",
            "familiar_brain.audio_tap",
            "--listen-host",
            "127.0.0.1",
            "--listen-port",
            str(listen_port),
            "--upstream-host",
            "127.0.0.1",
            "--upstream-port",
            str(upstream_port),
            "--control-host",
            "127.0.0.1",
            "--control-port",
            str(control_port),
            "--config",
            str(cfg_path),
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                out, err = self.proc.communicate(timeout=1.0)
                raise RuntimeError(f"audio_tap exited early ({self.proc.returncode}): {out}\n{err}")
            try:
                r = requests.get(f"{self.control_url}/_tap/current", timeout=0.3)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                time.sleep(0.05)
        self.close()
        raise TimeoutError("audio_tap did not become ready on /_tap/current within 8s")

    def ws_client(self, *, headers: dict[str, str] | None = None) -> WsRealtimeClient:
        return WsRealtimeClient("127.0.0.1", self.listen_port, "/v1/realtime", headers=headers)

    def current(self) -> dict[str, Any]:
        r = requests.get(f"{self.control_url}/_tap/current", timeout=2.0)
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2.0)


class StackHandle:
    """Opaque-box full stack handle running isolated temp files + real loopback HTTP servers."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.notes_dir = tmp_path / "notes"
        self.homebrew_dir = self.notes_dir / "homebrew"
        self.rules_dir = tmp_path / "rules"
        self.sessions_dir = tmp_path / "sessions"
        self.db_path = tmp_path / "familiar.db"
        self.persona_path = tmp_path / "persona.yaml"
        self.lexicon_path = tmp_path / "lexicon.tsv"
        self.players_path = tmp_path / "players.txt"
        self.speakers_path = tmp_path / "speakers.json"
        self.cfg_path = tmp_path / "brain.toml"

        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.homebrew_dir.mkdir(parents=True, exist_ok=True)
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self.persona_path.write_text(DEFAULT_PERSONA_YAML, encoding="utf-8")
        self.lexicon_path.write_text(
            "# Canonical\tvariant|variant\n"
            "Rocky\tRockie|Rocki|Roki|Rockey\n"
            "Brokk\tBrock\n"
            "Trenevain\tTrinivane|Trinovane\n"
            "Brokk Trenevain\tBrock Trinivane\n"
            "Krag\tCraig|Crag\n"
            "Caldus\tCalvis|Caldis\n"
            "Nephria\tNepheria\n"
            "Till\tLil Ventil\n"
            "Jade Halls\tjade hall|jade halls\n"
            "Gloamhaunt\tGloam haunt\n"
            "Vacoran Amulet\tVacoran amulet\n"
            "Inkdrinker\tInk Drinker\n",
            encoding="utf-8",
        )
        self.players_path.write_text("James\nElias\nTJ\n", encoding="utf-8")

        self.llm = FakeUpstreamLLMServer()
        self.s2s = FakeS2SServer()
        self.reachy = FakeReachyServer()
        self.tap_control_port = find_free_port()
        self._audio_taps: list[AudioTapProcess] = []
        self._server = None
        self._server_thread: threading.Thread | None = None
        self.brain_port: int = 0
        self.brain_url: str = ""
        self.client: HeadlessClient

        monkeypatch.setenv("FAMILIAR_EMBEDDER", "deterministic")
        monkeypatch.setenv("FAMILIAR_NO_ROBOT", "1")
        self.write_config()
        self.start_brain()

    def write_config(
        self,
        *,
        classify_timeout_s: float = 0.6,
        follow_up_window_s: float = 8.0,
        max_follow_ups: int = 3,
        min_gesture_gap_s: float = 6.0,
        reaction_gap_s: float = 10.0,
        mode_timeout_s: float = 300.0,
        speakers_enabled: bool = True,
        speaker_threshold: float = 0.72,
    ) -> None:
        self.cfg_path.write_text(
            f"""[server]
host = "127.0.0.1"
port = 0

[upstream]
base_url = "{self.llm.base_url}"
generate_slot = 0
classify_slot = 1
classify_timeout_s = {classify_timeout_s}

[gate]
follow_up_window_s = {follow_up_window_s}
max_follow_ups = {max_follow_ups}

[paths]
persona = "{self.persona_path.as_posix()}"
sessions_dir = "{self.sessions_dir.as_posix()}"
lexicon = "{self.lexicon_path.as_posix()}"
players = "{self.players_path.as_posix()}"

[knowledge]
db_path = "{self.db_path.as_posix()}"
notes_dir = "{self.notes_dir.as_posix()}"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7

[motion]
min_gesture_gap_s = {min_gesture_gap_s}
reaction_gap_s = {reaction_gap_s}
mode_timeout_s = {mode_timeout_s}

[speakers]
enabled = {"true" if speakers_enabled else "false"}
backend = "acoustic"
store_path = "{self.speakers_path.as_posix()}"
tap_url = "http://127.0.0.1:{self.tap_control_port}"
threshold = {speaker_threshold}
max_age_s = 15.0
""",
            encoding="utf-8",
        )

    def start_brain(self) -> None:
        app = create_app(self.cfg_path)
        self._server = make_server("127.0.0.1", 0, app, threaded=True)
        self.brain_port = int(self._server.server_port)
        self.brain_url = f"http://127.0.0.1:{self.brain_port}"
        self.s2s.brain_responses_url = f"{self.brain_url}/v1/responses"
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        self.client = HeadlessClient(self.brain_url)

    def stop_brain(self) -> None:
        if self._server is not None:
            try:
                state_r = requests.get(f"{self.brain_url}/api/state", timeout=1.0)
                if state_r.ok and state_r.json().get("session"):
                    requests.post(f"{self.brain_url}/api/session/end", json={}, timeout=2.0)
            except Exception:
                pass
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None

    def restart_brain(self) -> None:
        self.stop_brain()
        self.start_brain()

    def start_audio_tap(self, *, upstream_port: int | None = None) -> AudioTapProcess:
        listen_port = find_free_port()
        env = dict(os.environ)
        env["FAMILIAR_NO_ROBOT"] = "1"
        env["FAMILIAR_EMBEDDER"] = "deterministic"
        tap = AudioTapProcess(
            listen_port=listen_port,
            upstream_port=upstream_port or self.s2s.port,
            control_port=self.tap_control_port,
            cfg_path=self.cfg_path,
            env=env,
        )
        self._audio_taps.append(tap)
        return tap

    def copy_repo_notes(self) -> None:
        src_notes = REPO_ROOT / "data" / "notes"
        if src_notes.exists():
            shutil.copytree(src_notes, self.notes_dir, dirs_exist_ok=True)
        repo_lex = REPO_ROOT / "config" / "lexicon.tsv"
        if repo_lex.exists():
            shutil.copy2(repo_lex, self.lexicon_path)

    def close(self) -> None:
        for tap in self._audio_taps:
            tap.close()
        self._audio_taps.clear()
        self.stop_brain()
        self.reachy.close()
        self.s2s.close()
        self.llm.close()


@pytest.fixture
def fake_llm():
    srv = FakeUpstreamLLMServer()
    yield srv
    srv.close()


@pytest.fixture
def fake_s2s():
    srv = FakeS2SServer()
    yield srv
    srv.close()


@pytest.fixture
def fake_reachy():
    srv = FakeReachyServer()
    yield srv
    srv.close()


@pytest.fixture
def e2e_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    handle = StackHandle(tmp_path, monkeypatch)
    yield handle
    handle.close()


@pytest.fixture(scope="session")
def audio_samples() -> dict[str, Any]:
    brokk = brokk_pcm(2.0, 16000, seed=1)
    till = till_pcm(2.0, 16000, seed=2)
    dm = dm_pcm(2.0, 16000, seed=3)
    krag = krag_pcm(2.0, 16000, seed=4)
    stranger = stranger_pcm(2.0, 16000, seed=5)
    silence = silence_pcm(2.0, 16000)
    return {
        "brokk_pcm": brokk,
        "brokk_pcm_2": brokk_pcm(2.0, 16000, seed=11),
        "till_pcm": till,
        "dm_pcm": dm,
        "dm_pcm_2": dm_pcm(2.0, 16000, seed=13),
        "krag_pcm": krag,
        "stranger_pcm": stranger,
        "silence_pcm": silence,
        "short_pcm": brokk_pcm(0.15, 16000, seed=1),
        "brokk_b64": pcm16_to_b64(brokk),
        "till_b64": pcm16_to_b64(till),
        "dm_b64": pcm16_to_b64(dm),
        "krag_b64": pcm16_to_b64(krag),
        "stranger_b64": pcm16_to_b64(stranger),
        "silence_b64": pcm16_to_b64(silence),
        "dm_samples": pcm16_to_float_samples(dm),
        "brokk_samples": pcm16_to_float_samples(brokk),
        "silence_samples": pcm16_to_float_samples(silence),
    }


def to_bash_path(path: Path) -> str:
    """Convert a Windows path to a WSL /mnt/<drive>/... path when running bash on Windows."""
    s = str(path.resolve())
    if sys.platform == "win32" and len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return path.as_posix()
