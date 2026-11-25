"""Lightweight CPU-only ambient audio and background music analyzer.

Runs on incoming 16 kHz mono float32 audio inside audio_tap.py (0 MiB GPU VRAM, <2 ms/s
CPU). Classifies the room's non-verbal atmosphere so the native Reachy Mini app can drive
subtle, quiet creature posture and antenna behaviors (gentle sway during calm music, alert
tension during combat music, perk on sudden bursts, stillness during silence) without ever
triggering uninvited speech.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
MOODS = ("quiet", "calm_music", "tense_music", "combat_music", "lively_table", "sting")


@dataclass(frozen=True)
class AmbientSnapshot:
    mood: str = "quiet"
    energy: float = 0.0
    brightness: float = 0.0
    pulse: float = 0.0
    tempo_bpm: float = 0.0
    updated_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mood": self.mood,
            "energy": round(self.energy, 4),
            "brightness": round(self.brightness, 4),
            "pulse": round(self.pulse, 4),
            "tempo_bpm": round(self.tempo_bpm, 1),
            "updated_at": self.updated_at,
        }


class AmbientAudioAnalyzer:
    """Rolling acoustic atmosphere tracker for D&D table background music and dynamics."""

    def __init__(self, *, sample_rate: int = SAMPLE_RATE, window_frames: int = 24):
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._rms_hist: deque[float] = deque(maxlen=window_frames)
        self._flat_hist: deque[float] = deque(maxlen=window_frames)
        self._cent_hist: deque[float] = deque(maxlen=window_frames)
        self._low_hist: deque[float] = deque(maxlen=window_frames)
        self._onset_times: deque[float] = deque(maxlen=16)
        self._snapshot = AmbientSnapshot()

    def feed(self, pcm: np.ndarray, *, now: float | None = None) -> AmbientSnapshot:
        """Analyze one mono float32 chunk in [-1, 1] and update the rolling atmosphere."""
        if pcm is None or pcm.size < 64:
            return self.snapshot()
        ts = time.time() if now is None else now
        arr = np.asarray(pcm, dtype=np.float32).ravel()
        # Strip DC offset so hardware mic bias never inflates RMS or low-band energy.
        arr = arr - float(np.mean(arr))
        rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))

        # Spectral features via real FFT on up to 2048 samples (~128 ms at 16 kHz).
        frame = arr[-2048:] if arr.size > 2048 else arr
        windowed = frame * np.hanning(frame.size).astype(np.float32)
        mag_all = np.abs(np.fft.rfft(windowed)) + 1e-9
        freqs_all = np.fft.rfftfreq(frame.size, d=1.0 / self.sample_rate)
        # Ignore sub-75 Hz DC/AC/fan rumble so laptop fan hum doesn't look like combat drums.
        audible = freqs_all >= 75.0
        mag = mag_all[audible] if np.any(audible) else mag_all
        freqs = freqs_all[audible] if np.any(audible) else freqs_all

        # Spectral flatness: low (~0.01-0.25) for tonal music, higher for speech/noise.
        geo_mean = float(np.exp(np.mean(np.log(mag))))
        arith_mean = float(np.mean(mag))
        flatness = float(np.clip(geo_mean / (arith_mean + 1e-9), 0.0, 1.0))

        # Normalized spectral centroid (0..1 of Nyquist).
        centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-9))
        brightness = float(np.clip(centroid / (self.sample_rate * 0.5), 0.0, 1.0))

        # Low-frequency energy ratio (bass/drums below 300 Hz).
        total_pow = float(np.sum(mag * mag)) + 1e-12
        low_pow = float(np.sum((mag[freqs < 300.0]) ** 2))
        low_ratio = float(np.clip(low_pow / total_pow, 0.0, 1.0))

        with self._lock:
            prev_rms = float(np.mean(self._rms_hist)) if self._rms_hist else rms
            self._rms_hist.append(rms)
            self._flat_hist.append(flatness)
            self._cent_hist.append(brightness)
            self._low_hist.append(low_ratio)

            avg_rms = float(np.mean(self._rms_hist))
            avg_flat = float(np.mean(self._flat_hist))
            avg_bright = float(np.mean(self._cent_hist))
            avg_low = float(np.mean(self._low_hist))
            rms_std = float(np.std(self._rms_hist)) if len(self._rms_hist) >= 3 else 0.0
            pulse = float(np.clip(rms_std / (avg_rms + 1e-6), 0.0, 2.0))

            # Track transient peaks for rough tempo estimate (40..180 BPM).
            if rms > max(0.025, prev_rms * 1.45):
                if not self._onset_times or ts - self._onset_times[-1] >= 0.25:
                    self._onset_times.append(ts)
            while self._onset_times and ts - self._onset_times[0] > 8.0:
                self._onset_times.popleft()

            tempo_bpm = 0.0
            if len(self._onset_times) >= 3:
                diffs = np.diff(np.array(self._onset_times, dtype=np.float64))
                valid = diffs[(diffs >= 0.3) & (diffs <= 1.6)]
                if valid.size >= 2:
                    tempo_bpm = float(np.clip(60.0 / float(np.median(valid)), 40.0, 180.0))

            mood = self._classify(
                rms=rms,
                prev_rms=prev_rms,
                avg_rms=avg_rms,
                avg_flat=avg_flat,
                avg_bright=avg_bright,
                avg_low=avg_low,
                pulse=pulse,
                tempo_bpm=tempo_bpm,
            )
            self._snapshot = AmbientSnapshot(
                mood=mood,
                energy=min(1.0, avg_rms * 4.0),
                brightness=avg_bright,
                pulse=min(1.0, pulse),
                tempo_bpm=tempo_bpm,
                updated_at=ts,
            )
            return self._snapshot

    def _classify(
        self,
        *,
        rms: float,
        prev_rms: float,
        avg_rms: float,
        avg_flat: float,
        avg_bright: float,
        avg_low: float,
        pulse: float,
        tempo_bpm: float,
    ) -> str:
        # Sudden dramatic burst / sting / table cheer.
        if rms >= 0.14 and prev_rms > 1e-4 and rms >= prev_rms * 2.4:
            return "sting"
        if avg_rms < 0.012:
            return "quiet"
        # Tonal / sustained harmonic signal => background music.
        is_music = avg_flat < 0.34
        if is_music:
            if (avg_rms >= 0.055 and (pulse >= 0.28 or tempo_bpm >= 105.0)) or (
                avg_low >= 0.45 and avg_rms >= 0.03 and pulse >= 0.25
            ):
                return "combat_music"
            if avg_low >= 0.42 or avg_bright < 0.045:
                return "tense_music"
            return "calm_music"
        # Non-tonal active room sound (table chatter / dice / laughter).
        if avg_rms >= 0.045:
            return "lively_table"
        return "quiet"

    def snapshot(self) -> AmbientSnapshot:
        with self._lock:
            return self._snapshot
