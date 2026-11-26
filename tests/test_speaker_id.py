from __future__ import annotations

import asyncio
import base64
import json
import math
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import requests

from familiar_brain import app as app_mod
from familiar_brain import indexing
from familiar_brain.audio_tap import AudioTapProxy
from familiar_brain.config import Speakers, load_config
from familiar_brain.gate import Decision, Gate
from familiar_brain.motion import sleep_transition
from familiar_brain.retrieval import Store
from familiar_brain.session import SessionLog
from familiar_brain.speaker_id import (
    ACOUSTIC_DIM,
    AcousticEmbedder,
    InsufficientSpeechError,
    SpeakerRegistry,
    cosine_similarity,
    make_speaker_embedder,
)
from familiar_brain.state import BrainState
from tests.e2e.fakes import (
    FakeS2SServer,
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
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm_server import FakeLLM


def _make_brain_app(
    tmp_path: Path,
    *,
    fake: FakeLLM,
    monkeypatch,
    speakers_toml: str = "",
    gate_toml: str = "",
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = Store(tmp_path / "k.db", embedder=FakeEmbedder())
    monkeypatch.setattr(indexing, "open_store", lambda cfg: store)
    persona = tmp_path / "persona.yaml"
    persona.write_text(
        "name: Rocky\naliases: [rocky, rocki, rockie]\ntts_speaker: Aiden\nmanner: dry\n"
        "gestures:\n"
        "  attentive: play_emotion:attentive\n"
        "  amused: play_emotion:happy\n"
        "  alarmed: play_emotion:surprised\n"
        "  asleep: play_emotion:sleepy\n"
        "  awake: play_emotion:welcoming\n",
        encoding="utf-8",
    )
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "sessions").mkdir(exist_ok=True)
    (tmp_path / "players.txt").write_text("TJ\nJames\nElias\n", encoding="utf-8")
    cfg = tmp_path / "brain.toml"
    cfg.write_text(
        f"""
[server]
host = "127.0.0.1"
port = 0
[upstream]
base_url = "{fake.base_url}"
generate_slot = 0
classify_slot = 1
classify_timeout_s = 1.0
[gate]
follow_up_window_s = 8
{gate_toml}
[paths]
persona = "{persona.as_posix()}"
sessions_dir = "{(tmp_path / 'sessions').as_posix()}"
players = "{(tmp_path / 'players.txt').as_posix()}"
[knowledge]
db_path = "{(tmp_path / 'k.db').as_posix()}"
notes_dir = "{(tmp_path / 'notes').as_posix()}"
chunks_per_corpus = 3
max_context_chars = 4800
retention_days = 7
{speakers_toml}
""",
        encoding="utf-8",
    )
    app = app_mod.create_app(cfg)
    app.config["TESTING"] = True
    return app


def _turn(client, text: str, *, headers: dict[str, str] | None = None):
    resp = client.post(
        "/v1/responses",
        json={"model": "familiar", "stream": True, "tools": [], "input": [{"role": "user", "content": text}]},
        headers=headers or {},
    )
    resp.get_data()
    return resp


# ---------------------------------------------------------------------------
# AcousticEmbedder & OnnxSpeakerEmbedder (CPU-only)
# ---------------------------------------------------------------------------


def test_acoustic_embedder_discriminates_multiple_speakers_cpu_only():
    emb = AcousticEmbedder()
    assert emb.name == "acoustic" and emb.dim == ACOUSTIC_DIM == 124

    v_brokk_1 = emb.embed(brokk_pcm(2.0, 16000, seed=1), 16000)
    v_brokk_2 = emb.embed(pcm16_to_float_samples(brokk_pcm(2.0, 16000, seed=11)), 16000)
    v_dm = emb.embed(dm_pcm(2.0, 16000, seed=3), 16000)
    v_till = emb.embed(till_pcm(2.0, 16000, seed=2), 16000)
    v_krag = emb.embed(krag_pcm(2.0, 16000, seed=4), 16000)

    assert len(v_brokk_1) == 124
    assert math.isclose(sum(x * x for x in v_brokk_1), 1.0, rel_tol=1e-4)
    assert cosine_similarity(v_brokk_1, v_brokk_2) > 0.94
    assert cosine_similarity(v_brokk_1, v_dm) < 0.72
    assert cosine_similarity(v_brokk_1, v_till) < 0.72
    assert cosine_similarity(v_dm, v_till) < 0.72
    assert cosine_similarity(v_krag, v_till) < 0.72

    # 24 kHz audio is resampled cleanly and still matches Brokk
    v_brokk_24k = emb.embed(brokk_pcm(2.0, 24000, seed=7), 24000)
    assert cosine_similarity(v_brokk_1, v_brokk_24k) > 0.90


def test_acoustic_embedder_rejects_silence_and_short_clips():
    emb = AcousticEmbedder()
    with pytest.raises(InsufficientSpeechError):
        emb.embed(silence_pcm(2.0, 16000), 16000)
    with pytest.raises(InsufficientSpeechError):
        emb.embed(brokk_pcm(0.15, 16000, seed=1), 16000)
    with pytest.raises(InsufficientSpeechError):
        emb.embed(b"", 16000)


def test_make_speaker_embedder_auto_and_onnx_cpu(tmp_path, monkeypatch):
    # Auto falls back to AcousticEmbedder when onnxruntime/model absent
    assert isinstance(make_speaker_embedder("auto", onnx_model_path=tmp_path / "missing.onnx"), AcousticEmbedder)
    assert isinstance(make_speaker_embedder("acoustic"), AcousticEmbedder)
    with pytest.raises(FileNotFoundError):
        make_speaker_embedder("onnx", onnx_model_path=tmp_path / "missing.onnx")

    # Mock onnxruntime to verify CPUExecutionProvider + intra_op_num_threads=2
    model_file = tmp_path / "speaker.onnx"
    model_file.write_bytes(b"fake-onnx")
    captured: dict = {}

    class FakeSessionOptions:
        intra_op_num_threads = 0

    class FakeTensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class FakeInferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            captured["path"] = path
            captured["threads"] = sess_options.intra_op_num_threads
            captured["providers"] = providers

        def get_inputs(self):
            return [FakeTensor("feats", [1, None, 80])]

        def get_outputs(self):
            return [FakeTensor("emb", [1, 192])]

        def run(self, _out, feed):
            captured["shape"] = feed["feats"].shape
            return [np.ones((1, 192), dtype=np.float32)]

    fake_ort = types.SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        InferenceSession=FakeInferenceSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    emb = make_speaker_embedder("onnx", onnx_model_path=model_file)
    vec = emb.embed(dm_pcm(1.0, 16000), 16000)
    assert emb.name == "onnx" and emb.dim == 192
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["threads"] == 2
    assert captured["shape"][2] == 80
    assert math.isclose(sum(x * x for x in vec), 1.0, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# SpeakerRegistry
# ---------------------------------------------------------------------------


def test_speaker_registry_atomic_persistence_player_guard_and_matching(tmp_path):
    store_path = tmp_path / "speakers.json"
    reg = SpeakerRegistry(store_path, embedder=AcousticEmbedder())
    assert reg.enrolled_roles() == []

    # Reject empty / whitespace / real player names case-insensitively
    for bad in ("", "   ", "TJ", "tj", "James", "elias"):
        with pytest.raises(ValueError):
            reg.enroll(bad, dm_pcm(1.5, 16000), 16000, players=["TJ", "James", "Elias"])
    assert not store_path.exists()

    reg.enroll("DM", dm_pcm(2.0, 16000, seed=3), 16000, players=["TJ"])
    reg.enroll("Brokk", brokk_pcm(2.0, 16000, seed=1), 16000, players=["TJ"])
    reg.enroll("Till", till_pcm(2.0, 16000, seed=2), 16000, players=["TJ"])
    assert store_path.exists()
    assert not store_path.with_name("speakers.json.tmp").exists()

    # Reload from disk
    reg2 = SpeakerRegistry(store_path, embedder=AcousticEmbedder())
    assert reg2.enrolled_roles() == ["DM", "Brokk", "Till"]
    role, score = reg2.match(brokk_pcm(2.0, 16000, seed=11), 16000, threshold=0.72, min_margin=0.03)
    assert role == "Brokk" and score > 0.90

    # Silence and unknown speaker degrade cleanly to (None, ...)
    sil_role, _ = reg2.match(silence_pcm(1.5, 16000), 16000)
    assert sil_role is None
    str_role, _ = reg2.match(stranger_pcm(2.0, 16000, seed=5), 16000, threshold=0.72)
    assert str_role is None

    # Margin check rejects ambiguous tie
    assert reg2.match(brokk_pcm(2.0, 16000, seed=11), 16000, threshold=0.5, min_margin=0.99)[0] is None

    # Remove & clear
    assert reg2.remove("till") is True
    assert reg2.remove("nonexistent") is False
    assert reg2.enrolled_roles() == ["DM", "Brokk"]
    reg2.clear()
    assert reg2.enrolled_roles() == []


# ---------------------------------------------------------------------------
# AudioTapProxy (pure-stdlib asyncio WS/HTTP proxy + control endpoints)
# ---------------------------------------------------------------------------


def _start_proxy_thread(
    *,
    upstream_port: int,
    store_path: Path,
) -> tuple[AudioTapProxy, threading.Thread, asyncio.AbstractEventLoop]:
    listen_port = find_free_port()
    control_port = find_free_port()
    cfg = Speakers(
        enabled=True,
        backend="acoustic",
        store_path=str(store_path),
        tap_url=f"http://127.0.0.1:{control_port}",
        threshold=0.72,
    )
    proxy = AudioTapProxy(
        listen_host="127.0.0.1",
        listen_port=listen_port,
        upstream_host="127.0.0.1",
        upstream_port=upstream_port,
        control_host="127.0.0.1",
        control_port=control_port,
        speakers_cfg=cfg,
    )
    ready = threading.Event()
    loop_holder: list[asyncio.AbstractEventLoop] = []

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder.append(loop)
        loop.run_until_complete(proxy.start())
        ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(proxy.close())
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    assert ready.wait(timeout=5.0)
    return proxy, t, loop_holder[0]


def _stop_proxy_thread(t: threading.Thread, loop: asyncio.AbstractEventLoop) -> None:
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=3.0)


def test_audio_tap_proxy_http_ws_speaker_id_and_error_resilience(tmp_path):
    s2s = FakeS2SServer()
    store_path = tmp_path / "speakers.json"
    reg = SpeakerRegistry(store_path, embedder=AcousticEmbedder())
    reg.enroll("Brokk", brokk_pcm(2.0, 16000, seed=1), 16000)
    reg.enroll("DM", dm_pcm(2.0, 16000, seed=3), 16000)

    proxy, thread, loop = _start_proxy_thread(upstream_port=s2s.port, store_path=store_path)
    try:
        base = f"http://127.0.0.1:{proxy.listen_port}"
        ctrl = f"http://127.0.0.1:{proxy.control_port}"

        # HTTP proxy for /v1/pool and /v1/usage
        assert requests.get(f"{base}/v1/pool", timeout=2.0).json()["ready"] is True
        assert "connections" in requests.get(f"{base}/v1/usage", timeout=2.0).json()

        # WS relay + corrupt/empty audio resilience + speaker match + ping/pong
        with WsRealtimeClient("127.0.0.1", proxy.listen_port, "/v1/realtime") as ws:
            ws.send_json({"type": "session.update", "session": {"input_audio_sample_rate": 16000}})
            ws.send_json({"type": "input_audio_buffer.append", "audio": ""})
            ws.send_json({"type": "input_audio_buffer.append", "audio": "not!valid!b64"})
            ws.send_audio_pcm16(brokk_pcm(2.0, 16000, seed=11))
            ws.send_json({"type": "input_audio_buffer.commit"})

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                cur = requests.get(f"{ctrl}/_tap/current", timeout=1.0).json()
                if cur.get("role") == "Brokk":
                    break
                time.sleep(0.05)
            assert requests.get(f"{ctrl}/_tap/current", timeout=1.0).json()["role"] == "Brokk"

            # Silence clears speaker role on commit
            ws.send_audio_pcm16(silence_pcm(1.5, 16000))
            ws.send_json({"type": "input_audio_buffer.commit"})
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                cur = requests.get(f"{ctrl}/_tap/current", timeout=1.0).json()
                if cur.get("role") is None:
                    break
                time.sleep(0.05)
            assert requests.get(f"{ctrl}/_tap/current", timeout=1.0).json()["role"] is None

            # Server-to-client broadcast works
            s2s.broadcast_ws({"type": "response.text.delta", "delta": "Hi"})
            assert ws.recv_json(timeout=2.0)["delta"] == "Hi"
            assert ws.ping(b"sync") == b"sync"

        # /_tap/capture records live WS audio (with 24kHz session.update resampling)
        cap_box: list[requests.Response] = []

        def _cap():
            cap_box.append(requests.post(f"{ctrl}/_tap/capture", json={"duration_s": 0.5}, timeout=3.0))

        th = threading.Thread(target=_cap, daemon=True)
        th.start()
        time.sleep(0.1)
        with WsRealtimeClient("127.0.0.1", proxy.listen_port, "/v1/realtime") as ws2:
            ws2.send_json({"type": "session.update", "session": {"input_audio_sample_rate": 24000}})
            ws2.send_audio_pcm16(till_pcm(0.8, 24000, seed=2))
            time.sleep(0.55)
        th.join(timeout=3.0)
        assert cap_box and cap_box[0].status_code == 200
        cap_json = cap_box[0].json()
        assert cap_json["sample_rate"] == 16000 and cap_json["duration_s"] >= 0.4
        assert len(base64.b64decode(cap_json["audio_b64"])) > 0
    finally:
        _stop_proxy_thread(thread, loop)
        s2s.close()


def test_audio_tap_proxy_returns_502_when_upstream_down(tmp_path):
    dead_port = find_free_port()
    proxy, thread, loop = _start_proxy_thread(upstream_port=dead_port, store_path=tmp_path / "speakers.json")
    try:
        r = requests.get(f"http://127.0.0.1:{proxy.listen_port}/v1/pool", timeout=2.0)
        assert r.status_code == 502
    finally:
        _stop_proxy_thread(thread, loop)


# ---------------------------------------------------------------------------
# Gate, Deep Night, Persona & App Integration
# ---------------------------------------------------------------------------


def test_gate_dm_skips_followup_and_classifier():
    clf_calls: list = []

    def clf(segs):
        clf_calls.append(segs)
        return Decision.AMBIENT

    clock = {"t": 100.0}
    st = BrainState()
    g = Gate(state=st, name="Rocky", aliases=("rocky",), follow_up_window_s=8.0, classifier=clf, clock=lambda: clock["t"])

    assert g.decide(["Rocky, hello."], speaker_role="Brokk") is Decision.ADDRESSED
    clock["t"] += 2.0
    # DM inside follow-up window -> IGNORE, zero classifier calls
    assert g.decide(["The bridge groans under your boots."], speaker_role="DM") is Decision.IGNORE
    assert clf_calls == []
    # Explicit naming by DM still addresses Rocky
    assert g.decide(["Rocky, roll perception."], speaker_role="DM") is Decision.ADDRESSED
    # Player inside follow-up window still gets follow-up
    clock["t"] += 1.0
    assert g.decide(["What did you see?"], speaker_role="Brokk") is Decision.ADDRESSED


def test_gate_dm_narration_closes_active_followup_window():
    clf_calls: list = []

    def clf(segs):
        clf_calls.append(segs)
        return Decision.IGNORE

    clock = {"t": 100.0}
    st = BrainState()
    g = Gate(state=st, name="Rocky", aliases=("rocky",), follow_up_window_s=8.0, classifier=clf, clock=lambda: clock["t"])

    assert g.decide(["Rocky, ready?"], speaker_role="Brokk") is Decision.ADDRESSED
    assert st.last_named_at == 100.0

    clock["t"] += 1.0
    assert g.decide(["The alley is quiet."], speaker_role="DM") is Decision.IGNORE
    assert st.last_named_at is None
    assert clf_calls == []

    # Follow-up window is now closed; ambient unknown/player speech must not bleed through.
    clock["t"] += 1.0
    assert g.decide(["Pass the dice tray."]) is Decision.IGNORE
    assert g.decide(["Pass the dice tray."], speaker_role="Brokk") is Decision.IGNORE

    # Explicit DM re-addressing still opens a fresh follow-up window.
    clock["t"] += 1.0
    assert g.decide(["Rocky, roll stealth."], speaker_role="DM") is Decision.ADDRESSED
    assert st.last_named_at == 103.0
    clock["t"] += 1.0
    assert g.decide(["With advantage?"], speaker_role="Brokk") is Decision.ADDRESSED


def test_sleep_transition_phrases():
    assert sleep_transition("Deep night.") is True
    assert sleep_transition("Deep night falls over Nephria.") is True
    assert sleep_transition("It's deep night, everyone quiet.") is True
    assert sleep_transition("What happens during deep night?") is None
    assert sleep_transition("We must cross before deep night.") is None
    assert sleep_transition("Deep night is over.") is False
    assert sleep_transition("Deep night is over, Rocky.") is False
    assert sleep_transition("Deep night is over, Rocky, good morning.") is False
    assert sleep_transition("Good morning, Rocky!") is False


def test_speaker_apis_gating_persona_logging_and_fallback(tmp_path, monkeypatch):
    fake = FakeLLM()
    speakers_toml = f"""
[speakers]
enabled = true
backend = "acoustic"
store_path = "{(tmp_path / 'speakers.json').as_posix()}"
tap_url = "http://127.0.0.1:{find_free_port()}"
threshold = 0.72
max_age_s = 15.0
"""
    app = _make_brain_app(tmp_path, fake=fake, monkeypatch=monkeypatch, speakers_toml=speakers_toml)
    client = app.test_client()
    try:
        # Initial GET /api/speakers
        s0 = client.get("/api/speakers").get_json()
        assert s0["enabled"] is True and s0["enrolled"] == [] and s0["backend"] == "acoustic"

        # Zero-enrollment fallback: untagged turn behaves like baseline, no Current speaker block
        _turn(client, "Rocky, baseline check.")
        assert "Current speaker:" not in fake.requests[-1]["body"]["instructions"]

        # Reject invalid enrollments
        assert client.post("/api/speakers/enroll", json={"role": "  ", "audio_b64": pcm16_to_b64(dm_pcm())}).status_code == 400
        assert client.post("/api/speakers/enroll", json={"role": "tj", "audio_b64": pcm16_to_b64(dm_pcm())}).status_code == 400
        assert client.post("/api/speakers/enroll", json={"role": "DM", "audio_b64": pcm16_to_b64(silence_pcm())}).status_code == 400
        assert client.post("/api/speakers/enroll", json={"role": "DM", "audio_b64": pcm16_to_b64(brokk_pcm(0.15))}).status_code == 400
        # Live capture when tap unreachable -> 503
        assert client.post("/api/speakers/enroll", json={"role": "DM", "duration_s": 0.5}).status_code == 503

        # Enroll DM (samples) and Brokk (audio_b64)
        r_dm = client.post(
            "/api/speakers/enroll",
            json={"role": "DM", "samples": pcm16_to_float_samples(dm_pcm(2.0, 16000, seed=3)), "sample_rate": 16000},
        )
        assert r_dm.status_code == 200 and r_dm.get_json()["enrolled"] == ["DM"]
        r_brokk = client.post(
            "/api/speakers/enroll",
            json={"role": "Brokk", "audio_b64": pcm16_to_b64(brokk_pcm(2.0, 16000, seed=1)), "sample_rate": 16000},
        )
        assert r_brokk.status_code == 200 and set(r_brokk.get_json()["enrolled"]) == {"DM", "Brokk"}

        # Start session and check speaker tags + Deep Night DM authority + Brokk secrets
        sid = client.post("/api/session/start").get_json()["session"]
        _turn(client, "Rocky, what do I have?", headers={"X-Familiar-Speaker-Role": "Brokk", "X-Familiar-Speaker-Confidence": "not-a-float"})
        instr_brokk = fake.requests[-1]["body"]["instructions"]
        assert "Brokk Trenevain (your master and summoner)" in instr_brokk
        assert "Vacoran Amulet" in instr_brokk

        # DM inside follow-up window -> ignored, not talked over
        r_dm_narration = _turn(client, "The stone door seals shut.", headers={"X-Familiar-Speaker-Role": "DM"})
        assert "response.output_text.delta" not in r_dm_narration.get_data(as_text=True)

        # Non-DM player saying "Deep night." does NOT trigger Deep Night
        _turn(client, "Deep night.", headers={"X-Familiar-Speaker-Role": "Brokk"})
        assert client.get("/api/state").get_json()["asleep"] is False

        # Unrecognized untagged speaker while profiles are enrolled also does NOT trigger Deep Night
        _turn(client, "Deep night.")
        assert client.get("/api/state").get_json()["asleep"] is False

        # DM saying "Deep night falls over Nephria." DOES trigger Deep Night
        _turn(client, "Deep night falls over Nephria.", headers={"X-Familiar-Speaker-Role": "DM"})
        assert client.get("/api/state").get_json()["asleep"] is True
        _turn(client, "Deep night is over, Rocky.", headers={"X-Familiar-Speaker-Role": "DM"})
        assert client.get("/api/state").get_json()["asleep"] is False

        # Party companion persona context & session log speaker tags
        _turn(client, "Rocky, cover the door!", headers={"X-Familiar-Speaker-Role": "Krag", "X-Familiar-Speaker-Confidence": "0.88"})
        assert "Krag (a party companion" in fake.requests[-1]["body"]["instructions"]
        end_path = Path(client.post("/api/session/end").get_json()["path"])
        assert sid in end_path.name
        rows = [json.loads(ln) for ln in end_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert any(r.get("speaker") == "Krag" for r in rows)
        assert any(r.get("speaker") == "DM" for r in rows)

        # Delete & clear
        d1 = client.delete("/api/speakers/Brokk")
        assert d1.status_code == 200 and d1.get_json()["removed"] is True and d1.get_json()["enrolled"] == ["DM"]
        d2 = client.delete("/api/speakers")
        assert d2.status_code == 200 and d2.get_json()["enrolled"] == []
    finally:
        app.extensions["close"]()
        fake.close()


def test_checkpoint_toctou_and_main_warmup(tmp_path, monkeypatch):
    fake = FakeLLM()
    app = _make_brain_app(tmp_path, fake=fake, monkeypatch=monkeypatch)
    client = app.test_client()
    try:
        # Simulate TOCTOU where active_id exists during check, then snapshot_text raises RuntimeError
        client.post("/api/session/start")
        log = app.extensions["session_log"]
        monkeypatch.setattr(log, "snapshot_text", lambda: (_ for _ in ()).throw(RuntimeError("no active session")))
        r = client.post("/api/session/checkpoint", json={"force": True})
        assert r.status_code == 409 and r.get_json() == {"ok": False, "error": "no active session"}
    finally:
        app.extensions["close"]()
        fake.close()

    # Verify main() pre-warms context and (with FAMILIAR_WARMUP_LLM=1) runs streaming warmup cleanly
    fake2 = FakeLLM()
    app2 = _make_brain_app(tmp_path / "m2", fake=fake2, monkeypatch=monkeypatch)
    ran: dict = {}

    class DummyApp:
        extensions = app2.extensions
        logger = app2.logger

        def run(self, host, port, threaded=True):
            ran["host"] = host
            ran["port"] = port
            st_inside = app2.extensions["brain_state"]
            assert st_inside.recent_transcript() == []
            assert st_inside.recent_segments == []
            assert st_inside.last_named_at is None
            assert st_inside.last_spoke_at is None
            assert st_inside.last_turn_ms is None
            assert st_inside.tracking is False
            assert app2.extensions["session_log"].active_id is None

    monkeypatch.setenv("FAMILIAR_WARMUP_LLM", "1")
    monkeypatch.setattr(app_mod, "create_app", lambda _cfg: DummyApp())
    monkeypatch.setattr(app_mod, "load_config", lambda _cfg: load_config(tmp_path / "m2" / "brain.toml"))
    monkeypatch.setattr(sys, "argv", ["familiar_brain", str(tmp_path / "m2" / "brain.toml")])
    try:
        app_mod.main()
        assert ran["host"] == "127.0.0.1"
        resps = [r for r in fake2.requests if r["path"].endswith("/responses")]
        assert resps and resps[-1]["body"]["stream"] is True
        assert "tools" in resps[-1]["body"] and any(
            t.get("name") == "roll" for t in resps[-1]["body"]["tools"]
        )
        assert list((tmp_path / "m2" / "sessions").glob("*.jsonl")) == []
    finally:
        app2.extensions["close"]()
        fake2.close()


def test_session_log_speaker_tags(tmp_path):
    log = SessionLog(tmp_path / "sessions")
    log.start()
    log.append(text="Roll initiative.", decision="IGNORE", spoke=False, speaker="DM", speaker_confidence=0.91234)
    log.append(text="Wait.", decision="IGNORE", spoke=False)
    rows = [json.loads(ln) for ln in log.snapshot_text().splitlines() if ln.strip()]
    assert rows[0]["speaker"] == "DM" and rows[0]["speaker_confidence"] == 0.9123
    assert "speaker" not in rows[1] and "speaker_confidence" not in rows[1]


def test_tap_reload_clears_removed_role_and_brain_reads_tap_state(tmp_path, monkeypatch):
    import familiar_brain.speaker_id as sid_mod

    assert "torch" not in sid_mod.__dict__

    s2s = FakeS2SServer()
    fake = FakeLLM()
    store_path = tmp_path / "speakers.json"
    proxy, thread, loop = _start_proxy_thread(upstream_port=s2s.port, store_path=store_path)
    speakers_toml = f"""
[speakers]
enabled = true
backend = "acoustic"
store_path = "{store_path.as_posix()}"
tap_url = "http://127.0.0.1:{proxy.control_port}"
threshold = 0.72
max_age_s = 15.0
"""
    app = _make_brain_app(tmp_path, fake=fake, monkeypatch=monkeypatch, speakers_toml=speakers_toml)
    client = app.test_client()
    try:
        # Enroll DM and Brokk via HTTP API (notifies /_tap/reload)
        assert client.post(
            "/api/speakers/enroll",
            json={"role": "DM", "audio_b64": pcm16_to_b64(dm_pcm(2.0, 16000, seed=3)), "sample_rate": 16000},
        ).status_code == 200
        assert client.post(
            "/api/speakers/enroll",
            json={"role": "Brokk", "audio_b64": pcm16_to_b64(brokk_pcm(2.0, 16000, seed=1)), "sample_rate": 16000},
        ).status_code == 200

        # Stream Brokk audio over WS without setting X-Familiar-Speaker-Role header on /v1/responses
        with WsRealtimeClient("127.0.0.1", proxy.listen_port, "/v1/realtime") as ws:
            ws.send_audio_pcm16(brokk_pcm(2.0, 16000, seed=11))
            ws.send_json({"type": "input_audio_buffer.commit"})
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                cur = requests.get(f"http://127.0.0.1:{proxy.control_port}/_tap/current", timeout=1.0).json()
                if cur.get("role") == "Brokk":
                    break
                time.sleep(0.05)

        client.post("/api/session/start")
        _turn(client, "Rocky, check my gear.")
        instr = fake.requests[-1]["body"]["instructions"]
        assert "Brokk Trenevain (your master and summoner)" in instr

        st = client.get("/api/state").get_json()
        assert st["speakers"]["enabled"] is True
        assert set(st["speakers"]["enrolled"]) == {"DM", "Brokk"}
        assert st["speakers"]["last_speaker"] == "Brokk"
        assert isinstance(st["speakers"]["last_confidence"], float)
        assert st["speakers"]["capturing"] is False
        assert st["transcript"] and st["transcript"][-1]["speaker"] == "Brokk"

        # Removing active speaker role triggers /_tap/reload and clears stale tap role
        assert client.delete("/api/speakers/Brokk").status_code == 200
        cur_after = requests.get(f"http://127.0.0.1:{proxy.control_port}/_tap/current", timeout=1.0).json()
        assert cur_after["role"] is None
    finally:
        app.extensions["close"]()
        _stop_proxy_thread(thread, loop)
        s2s.close()
        fake.close()


def test_api_warmup_isolates_brain_state_session_log_and_timing_logs(tmp_path, monkeypatch, caplog):
    import logging

    fake = FakeLLM()
    app = _make_brain_app(tmp_path, fake=fake, monkeypatch=monkeypatch)
    client = app.test_client()
    try:
        # Invalid payloads -> 400
        assert client.post("/api/warmup", json=["bad"]).status_code == 400
        assert client.post("/api/warmup", json={"llm": "yes"}).status_code == 400

        # Start an active session to prove warmup never writes to it either
        client.post("/api/session/start")
        roll_call = {
            "id": "fc_w1",
            "type": "function_call",
            "call_id": "call_w1",
            "name": "roll",
            "arguments": '{"expression":"1d20"}',
            "status": "completed",
        }
        fake.responses_events.append([
            {"type": "response.created", "response": {"id": "rw1", "output": []}},
            {"type": "response.output_item.done", "output_index": 0, "item": roll_call},
            {"type": "response.completed", "response": {"id": "rw1", "output": [roll_call]}},
        ])
        fake.responses_events.append([
            {"type": "response.output_text.delta", "delta": "Rolled a fourteen."},
            {"type": "response.completed", "response": {"id": "rw2", "output": []}},
        ])

        with caplog.at_level(logging.INFO, logger="familiar.timing"):
            r_warm = client.post("/api/warmup", json={"llm": True})
        assert r_warm.status_code == 200
        body = r_warm.get_json()
        assert body["ok"] is True and body["llm"] is True and body["frames"] > 0

        # Also test non-LLM warmup
        r_no_llm = client.post("/api/warmup", json={"llm": False})
        assert r_no_llm.status_code == 200 and r_no_llm.get_json()["llm"] is False

        st = app.extensions["brain_state"]
        assert st.recent_transcript() == []
        assert st.recent_segments == []
        assert st.last_named_at is None
        assert st.last_spoke_at is None
        assert st.last_turn_ms is None
        assert st.tracking is False
        assert app.extensions["session_log"].snapshot_text() == ""
        timing_msgs = [rec.getMessage() for rec in caplog.records if rec.name == "familiar.timing"]
        assert timing_msgs and all('"decision": "ADDRESSED"' not in m for m in timing_msgs)
        assert any('"decision": "WARMUP"' in m for m in timing_msgs)

        # Unreachable upstream during warmup degrades gracefully without failing or polluting state
        app.extensions["upstream"].base_url = f"http://127.0.0.1:{find_free_port()}/v1"
        app.extensions["upstream"].timeout_s = 1.0
        r_down = client.post("/api/warmup", json={"llm": True})
        assert r_down.status_code == 200 and r_down.get_json()["llm"] is False
        assert st.recent_transcript() == [] and st.tracking is False
    finally:
        app.extensions["close"]()
        fake.close()


def test_concurrent_speaker_enroll_capture_count_and_409(tmp_path, monkeypatch):
    fake = FakeLLM()
    speakers_toml = f"""
[speakers]
enabled = true
backend = "acoustic"
store_path = "{(tmp_path / 'speakers.json').as_posix()}"
tap_url = "http://127.0.0.1:{find_free_port()}"
threshold = 0.72
max_age_s = 15.0
"""
    app = _make_brain_app(tmp_path, fake=fake, monkeypatch=monkeypatch, speakers_toml=speakers_toml)

    entered = threading.Event()
    release = threading.Event()
    orig_post = requests.Session.post
    capture_payload = {
        "audio_b64": pcm16_to_b64(dm_pcm(1.5, 16000, seed=3)),
        "sample_rate": 16000,
    }

    class FakeResp:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self.ok = 200 <= status_code < 300
            self._payload = payload

        def json(self):
            return self._payload

    mode = {"status": 200}

    def fake_session_post(self, url, *args, **kwargs):
        if str(url).endswith("/_tap/capture"):
            entered.set()
            release.wait(timeout=5.0)
            return FakeResp(mode["status"], capture_payload)
        return orig_post(self, url, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "post", fake_session_post)

    try:
        r1_holder: dict = {}

        def run_enroll_1():
            r = app.test_client().post(
                "/api/speakers/enroll",
                json={"role": "DM", "duration_s": 1.0},
            )
            r1_holder["status"] = r.status_code
            r1_holder["body"] = r.get_json()

        t1 = threading.Thread(target=run_enroll_1)
        t1.start()
        assert entered.wait(timeout=5.0)

        c2 = app.test_client()
        assert c2.get("/api/speakers").get_json()["capturing"] is True
        assert c2.get("/api/state").get_json()["speakers"]["capturing"] is True

        # Overlapping enroll gets early 409 and MUST NOT drop capturing=True for Request 1
        r2 = c2.post("/api/speakers/enroll", json={"role": "Brokk", "duration_s": 1.0})
        assert r2.status_code == 409
        assert c2.get("/api/speakers").get_json()["capturing"] is True
        assert c2.get("/api/state").get_json()["speakers"]["capturing"] is True

        release.set()
        t1.join(timeout=5.0)
        assert r1_holder["status"] == 200 and r1_holder["body"]["ok"] is True
        assert c2.get("/api/speakers").get_json()["capturing"] is False
        assert c2.get("/api/state").get_json()["speakers"]["capturing"] is False

        # Upstream 409 from /_tap/capture is passed through as 409 (not 503) and leaves capturing=False
        entered.clear()
        mode["status"] = 409
        r_ext = c2.post("/api/speakers/enroll", json={"role": "Krag", "duration_s": 1.0})
        assert r_ext.status_code == 409
        assert c2.get("/api/speakers").get_json()["capturing"] is False
    finally:
        release.set()
        app.extensions["close"]()
        fake.close()




def _bare_tap(tmp_path: Path) -> AudioTapProxy:
    cfg = Speakers(enabled=True, backend="acoustic", store_path=str(tmp_path / "speakers.json"))
    proxy = AudioTapProxy(speakers_cfg=cfg)
    proxy.registry.profiles = {"DM": {}}
    return proxy


def _append_msg(seconds: float, sr: int = 16000) -> bytes:
    pcm = (np.sin(np.linspace(0, 440 * 2 * math.pi * seconds, int(sr * seconds))) * 8000).astype("<i2")
    return json.dumps(
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm.tobytes()).decode("ascii")}
    ).encode("utf-8")


def test_tap_role_clears_on_streaming_miss(tmp_path):
    # Real clients never send input_audio_buffer.commit, so a streaming miss must clear the DM label.
    proxy = _bare_tap(tmp_path)
    results = iter([("DM", 0.9), ("DM", 0.9)])
    proxy.registry.match = lambda *a, **k: next(results, (None, 0.3))

    async def run() -> None:
        proxy._wake_event = asyncio.Event()
        worker = asyncio.create_task(proxy._tap_worker())
        state = {"sr": 16000}
        try:
            for _ in range(6):
                proxy._inspect_client_text(_append_msg(0.5), state)
                await asyncio.sleep(0.45)
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker

    asyncio.run(run())
    assert proxy.current_state()["role"] is None


def _stream_roles(tmp_path, results, windows: int):
    proxy = _bare_tap(tmp_path)
    it = iter(results)
    proxy.registry.match = lambda *a, **k: next(it, ("DM", 0.9))

    async def run() -> None:
        proxy._wake_event = asyncio.Event()
        worker = asyncio.create_task(proxy._tap_worker())
        state = {"sr": 16000}
        try:
            for _ in range(windows):
                proxy._inspect_client_text(_append_msg(0.5), state)
                await asyncio.sleep(0.45)
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker

    asyncio.run(run())
    return proxy.current_state()["role"]


def test_one_noisy_streaming_window_keeps_the_dm_label(tmp_path):
    # A laugh or dice clatter in one window must not drop the DM mid-turn.
    assert _stream_roles(tmp_path, [("DM", 0.9), (None, 0.3)], windows=2) == "DM"


def test_silent_streaming_window_keeps_the_dm_label(tmp_path):
    # No voice in the window (conf 0): leave expiry to speakers.max_age_s.
    assert _stream_roles(tmp_path, [("DM", 0.9), (None, 0.0), (None, 0.0)], windows=3) == "DM"


def test_startup_tap_sync_outlasts_a_cold_stack_start():
    # start_all.sh starts the tap only after llama (up to 600 s) and s2s (up to 240 s) are ready.
    import inspect
    import re

    from familiar_brain import app as app_mod

    src = inspect.getsource(app_mod)
    m = re.search(r"def _sync_tap_store\(\*, attempts: int = (\d+), interval_s: float = ([\d.]+)\)", src)
    assert m and int(m.group(1)) * float(m.group(2)) >= 900


def test_tap_worker_survives_exception(tmp_path, caplog):
    import logging

    proxy = _bare_tap(tmp_path)
    calls = {"n": 0}

    def flaky_match(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return ("DM", 0.9)

    proxy.registry.match = flaky_match
    commit = json.dumps({"type": "input_audio_buffer.commit"}).encode("utf-8")

    async def run() -> bool:
        proxy._wake_event = asyncio.Event()
        worker = asyncio.create_task(proxy._tap_worker())
        state = {"sr": 16000}
        try:
            proxy._inspect_client_text(_append_msg(1.0), state)
            proxy._inspect_client_text(commit, state)
            await asyncio.sleep(0.5)
            alive = not worker.done()
            proxy._inspect_client_text(_append_msg(1.0), state)
            proxy._inspect_client_text(commit, state)
            await asyncio.sleep(0.5)
            return alive and not worker.done()
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker

    with caplog.at_level(logging.WARNING, logger="familiar.audio_tap"):
        alive = asyncio.run(run())
    assert alive
    assert any(r.levelno >= logging.WARNING and "boom" in r.getMessage() + str(r.exc_info) for r in caplog.records)
    assert calls["n"] == 2
    assert proxy.current_state()["role"] == "DM"


def test_tap_reads_ga_input_rate(tmp_path):
    proxy = _bare_tap(tmp_path)
    state = {"sr": 16000}
    msg = {
        "type": "session.update",
        "session": {"type": "realtime", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}},
    }
    proxy._inspect_client_text(json.dumps(msg).encode("utf-8"), state)
    assert state["sr"] == 24000


def _capture_tap_posts(monkeypatch, *, ok_after: int = 0) -> list[dict]:
    posts: list[dict] = []

    class FakeResp:
        status_code = 200
        ok = True

    def fake_post(self, url, *args, **kwargs):
        if str(url).endswith("/_tap/reload"):
            posts.append(kwargs.get("json") or {})
            if len(posts) <= ok_after:
                raise requests.ConnectionError("tap not up yet")
            return FakeResp()
        raise AssertionError(f"unexpected post {url}")

    monkeypatch.setattr(requests.Session, "post", fake_post)
    return posts


def _speaker_app(tmp_path, monkeypatch, fake):
    speakers_toml = f"""
[speakers]
enabled = true
backend = "acoustic"
store_path = "{(tmp_path / 'speakers.json').as_posix()}"
tap_url = "http://127.0.0.1:{find_free_port()}"
threshold = 0.72
max_age_s = 3.0
"""
    return _make_brain_app(tmp_path, fake=fake, monkeypatch=monkeypatch, speakers_toml=speakers_toml)


def test_notify_tap_reload_always_sends_store_path(tmp_path, monkeypatch):
    fake = FakeLLM()
    app = _speaker_app(tmp_path, monkeypatch, fake)
    posts = _capture_tap_posts(monkeypatch)
    try:
        assert app.test_client().delete("/api/speakers/DM").status_code == 200
        assert posts, "delete should notify the tap"
        expected = str(app.extensions["speaker_registry"].store_path)
        assert posts[-1].get("store_path") == expected
    finally:
        app.extensions["close"]()
        fake.close()


def test_tap_store_sync_retries_in_background_until_tap_answers(tmp_path, monkeypatch):
    fake = FakeLLM()
    app = _speaker_app(tmp_path, monkeypatch, fake)
    posts = _capture_tap_posts(monkeypatch, ok_after=2)
    try:
        t0 = time.monotonic()
        t = app.extensions["sync_tap_store"](attempts=10, interval_s=0.05)
        assert time.monotonic() - t0 < 0.5, "startup sync must not block the caller"
        assert t.daemon
        t.join(timeout=5.0)
        assert not t.is_alive()
        assert len(posts) == 3, "stops retrying once the tap answers"
        expected = str(app.extensions["speaker_registry"].store_path)
        assert all(p.get("store_path") == expected for p in posts)
    finally:
        app.extensions["close"]()
        fake.close()


def test_tap_store_sync_gives_up_with_warning(tmp_path, monkeypatch, caplog):
    import logging

    fake = FakeLLM()
    app = _speaker_app(tmp_path, monkeypatch, fake)
    posts = _capture_tap_posts(monkeypatch, ok_after=100)
    try:
        with caplog.at_level(logging.WARNING):
            t = app.extensions["sync_tap_store"](attempts=3, interval_s=0.01)
            t.join(timeout=5.0)
        assert not t.is_alive()
        assert len(posts) == 3
        assert any(r.levelno >= logging.WARNING and "_tap/reload" in r.getMessage() for r in caplog.records)
    finally:
        app.extensions["close"]()
        fake.close()
