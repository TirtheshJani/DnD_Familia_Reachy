from __future__ import annotations

import asyncio
import json

import numpy as np

from familiar_brain.ambient_audio import SAMPLE_RATE, AmbientAudioAnalyzer
from familiar_brain.audio_tap import AudioTapProxy
from familiar_brain.config import Speakers


def _sine_chord(freqs: tuple[float, ...], seconds: float = 0.25, amp: float = 0.08) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    sig = sum(np.sin(2.0 * np.pi * f * t) for f in freqs) / max(1, len(freqs))
    return (sig * amp).astype(np.float32)


def test_ambient_analyzer_distinguishes_quiet_music_combat_and_sting():
    az = AmbientAudioAnalyzer(sample_rate=SAMPLE_RATE)

    # Silence -> quiet
    silence = np.zeros(4000, dtype=np.float32)
    snap = az.feed(silence, now=1.0)
    assert snap.mood == "quiet"

    # Calm harmonic tavern chord -> calm_music
    calm_az = AmbientAudioAnalyzer(sample_rate=SAMPLE_RATE)
    chord = _sine_chord((440.0, 554.37, 659.25), amp=0.05)
    for i in range(6):
        snap = calm_az.feed(chord, now=1.0 + i * 0.25)
    assert snap.mood == "calm_music"

    # Low bass-heavy combat music -> combat_music
    combat_az = AmbientAudioAnalyzer(sample_rate=SAMPLE_RATE)
    bass_loud = _sine_chord((80.0, 120.0, 160.0), amp=0.14)
    bass_soft = _sine_chord((80.0, 120.0, 160.0), amp=0.04)
    for i in range(8):
        chunk = bass_loud if i % 2 == 0 else bass_soft
        snap = combat_az.feed(chunk, now=1.0 + i * 0.4)
    assert snap.mood == "combat_music"

    # Sudden loud transient over quiet bed -> sting
    sting_az = AmbientAudioAnalyzer(sample_rate=SAMPLE_RATE)
    for i in range(5):
        sting_az.feed(_sine_chord((440.0,), amp=0.02), now=1.0 + i * 0.2)
    burst = _sine_chord((320.0, 640.0, 960.0), amp=0.35)
    snap = sting_az.feed(burst, now=2.5)
    assert snap.mood == "sting"


def test_audio_tap_control_exposes_ambient_endpoint(tmp_path):
    async def run() -> dict:
        cfg = Speakers(enabled=True, store_path=str(tmp_path / "speakers.json"))
        proxy = AudioTapProxy(
            listen_host="127.0.0.1",
            listen_port=0,
            upstream_host="127.0.0.1",
            upstream_port=9,
            control_host="127.0.0.1",
            control_port=0,
            speakers_cfg=cfg,
        )
        await proxy.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy.control_port)
            writer.write(b"GET /_tap/ambient HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            await writer.drain()
            raw = await reader.read()
            writer.close()
            body = raw.split(b"\r\n\r\n", 1)[1]
            return json.loads(body.decode("utf-8"))
        finally:
            await proxy.close()

    data = asyncio.run(run())
    assert data["mood"] == "quiet"
    assert "energy" in data
