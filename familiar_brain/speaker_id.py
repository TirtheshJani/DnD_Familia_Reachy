"""CPU-only speaker recognition: acoustic/ONNX embedders and role profile registry."""
from __future__ import annotations

import importlib.util
import json
import math
import threading
import time
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly

from familiar_brain.summariser import write_atomic

TARGET_SAMPLE_RATE = 16000
WIN_LEN = 400  # 25 ms at 16 kHz
HOP_LEN = 160  # 10 ms at 16 kHz
N_FFT = 512
N_MELS = 40
N_MFCC = 19  # c1..c19 (c0 dropped to ignore mic distance/gain)
MIN_ACTIVE_FRAMES = 40  # >= 0.40 s voiced speech
ACOUSTIC_DIM = (N_MFCC * 3 + 5) * 2  # (19 + 19 + 19 + 5) * [mean, std] = 124

_HAMMING = np.hamming(WIN_LEN).astype(np.float32)
_FREQS = np.fft.rfftfreq(N_FFT, 1.0 / TARGET_SAMPLE_RATE).astype(np.float32)
_LAG_MIN = int(TARGET_SAMPLE_RATE // 400)  # 40 samples -> 400 Hz
_LAG_MAX = int(TARGET_SAMPLE_RATE // 65)   # 246 samples -> 65 Hz
_LAGS = np.arange(_LAG_MIN, _LAG_MAX + 1, dtype=np.int32)
_LOG2_F0_REF = math.log2(150.0)


def _build_mel_filterbank(
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    sample_rate: int = TARGET_SAMPLE_RATE,
    f_min: float = 80.0,
    f_max: float = 7600.0,
) -> np.ndarray:
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts = np.array([mel_to_hz(float(m)) for m in mel_pts], dtype=np.float64)
    bins = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        if center > left:
            for k in range(left, center):
                fb[m - 1, k] = (k - left) / (center - left)
        if right > center:
            for k in range(center, right):
                fb[m - 1, k] = (right - k) / (right - center)
        if center == left and right == center and center < fb.shape[1]:
            fb[m - 1, center] = 1.0
    return fb


_MEL_FB_40 = _build_mel_filterbank(n_mels=40)
_MEL_FB_80 = _build_mel_filterbank(n_mels=80)


class InsufficientSpeechError(ValueError):
    """Raised when audio contains silence or < 0.40s of active voiced speech."""


class SpeakerEmbedder(Protocol):
    name: str
    dim: int

    def embed(self, pcm16_or_f32: bytes | np.ndarray | Sequence[float], sample_rate: int = 16000) -> list[float]: ...


def _to_mono_f32(audio: bytes | bytearray | memoryview | np.ndarray | Sequence[float], sample_rate: int = 16000) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if isinstance(audio, (bytes, bytearray, memoryview)):
        raw = bytes(audio)
        usable = len(raw) - (len(raw) % 2)
        if usable <= 0:
            return np.zeros(0, dtype=np.float32)
        x = np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32) / 32768.0
    else:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return np.zeros(0, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.mean(arr, axis=-1 if arr.shape[1] <= 8 else 0)
        x = arr.reshape(-1).astype(np.float32, copy=False)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    if sample_rate != TARGET_SAMPLE_RATE and len(x) > 0:
        g = math.gcd(TARGET_SAMPLE_RATE, int(sample_rate))
        up = TARGET_SAMPLE_RATE // g
        down = int(sample_rate) // g
        x = resample_poly(x, up, down).astype(np.float32, copy=False)
    return x


def _deltas(feat: np.ndarray, width: int = 2) -> np.ndarray:
    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    denom = 2.0 * sum(n * n for n in range(1, width + 1))
    out = np.zeros_like(feat, dtype=np.float32)
    t_len = feat.shape[0]
    for n in range(1, width + 1):
        out += n * (padded[width + n : width + n + t_len] - padded[width - n : width - n + t_len])
    return out / denom


def _active_frames(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < WIN_LEN:
        raise InsufficientSpeechError("audio clip shorter than one 25ms frame")
    frames = np.lib.stride_tricks.sliding_window_view(x, WIN_LEN)[::HOP_LEN]
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    thresh = max(0.008, 0.22 * float(np.percentile(rms, 90)))
    mask = rms >= thresh
    active_count = int(np.sum(mask))
    if active_count < MIN_ACTIVE_FRAMES:
        raise InsufficientSpeechError(
            f"insufficient active speech ({active_count * 0.01:.2f}s < {MIN_ACTIVE_FRAMES * 0.01:.2f}s)"
        )
    return frames, mask


def _l2_normalize(vec: np.ndarray) -> list[float]:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12 or not math.isfinite(norm):
        out = np.zeros_like(vec, dtype=np.float32)
        if len(out) > 0:
            out[0] = 1.0
        return out.tolist()
    return (vec / norm).astype(np.float32).tolist()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class AcousticEmbedder:
    """Zero-dependency pure numpy/scipy CPU-only speaker embedder (124-d L2-normalized)."""

    name = "acoustic"
    dim = ACOUSTIC_DIM

    def embed_with_duration(
        self,
        pcm16_or_f32: bytes | bytearray | memoryview | np.ndarray | Sequence[float],
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> tuple[list[float], float]:
        x = _to_mono_f32(pcm16_or_f32, sample_rate)
        frames, mask = _active_frames(x)
        active_s = float(np.sum(mask)) * (HOP_LEN / TARGET_SAMPLE_RATE)

        # Pre-emphasis + windowed FFT + 40-Mel log filterbank
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - 0.97 * x[:-1]
        y_frames = np.lib.stride_tricks.sliding_window_view(y, WIN_LEN)[::HOP_LEN] * _HAMMING
        spec = np.fft.rfft(y_frames, n=N_FFT, axis=1)
        power = (np.real(spec) ** 2 + np.imag(spec) ** 2).astype(np.float32)
        log_mel = np.log(np.maximum(power @ _MEL_FB_40.T, 1e-10))

        # 19 static MFCCs c1..c19 (c0 dropped) + delta + delta-delta
        mfcc = dct(log_mel, type=2, norm="ortho", axis=-1)[:, 1 : 1 + N_MFCC].astype(np.float32)
        d1 = _deltas(mfcc)
        d2 = _deltas(d1)

        mfcc_a = mfcc[mask]
        d1_a = d1[mask]
        d2_a = d2[mask]

        # Normalized autocorrelation F0 over active frames
        f_raw = frames[mask]
        f_centered = f_raw - np.mean(f_raw, axis=1, keepdims=True)
        f_spec = np.fft.rfft(f_centered, n=1024, axis=1)
        acf = np.fft.irfft(f_spec * np.conj(f_spec), n=1024, axis=1)[:, :WIN_LEN]
        cum = np.cumsum(f_centered * f_centered, axis=1)
        e1 = cum[:, WIN_LEN - 1 - _LAGS]
        e2 = cum[:, WIN_LEN - 1 : WIN_LEN] - cum[:, _LAGS - 1]
        nacf = acf[:, _LAGS] / np.sqrt(np.maximum(e1 * e2, 1e-12))
        best_idx = np.argmax(nacf, axis=1)
        best_val = nacf[np.arange(len(nacf)), best_idx]
        best_lag = _LAGS[best_idx]
        voiced = best_val >= 0.35
        if np.any(voiced):
            log2_f0_vals = (np.log2(TARGET_SAMPLE_RATE / best_lag[voiced]) - _LOG2_F0_REF) * 4.0
            f0_mean = float(np.mean(log2_f0_vals))
            f0_std = float(np.std(log2_f0_vals))
        else:
            f0_mean, f0_std = 0.0, 0.0

        # Spectral descriptors over active frames
        p_a = power[mask]
        mag = np.sqrt(p_a)
        mag_sum = np.sum(mag, axis=1) + 1e-12
        centroid = np.sum(mag * _FREQS[None, :], axis=1) / mag_sum
        bandwidth = np.sqrt(np.sum(mag * ((_FREQS[None, :] - centroid[:, None]) ** 2), axis=1) / mag_sum + 1e-12)
        flatness = np.exp(np.mean(np.log(p_a + 1e-12), axis=1)) / (np.mean(p_a, axis=1) + 1e-12)
        zcr = np.mean((f_raw[:, 1:] * f_raw[:, :-1]) < 0.0, axis=1)

        desc = np.stack(
            [
                (centroid - 1600.0) / 1000.0,
                (bandwidth - 1000.0) / 800.0,
                flatness * 4.0,
                (zcr - 0.10) * 10.0,
            ],
            axis=1,
        ).astype(np.float32)

        # Empirical scale weights so MFCCs, deltas, pitch, and spectral stats align cleanly
        seq_feats = np.concatenate([mfcc_a * 0.75, d1_a * 1.4, d2_a * 2.0, desc], axis=1)
        means = np.concatenate([np.mean(seq_feats, axis=0), np.array([f0_mean], dtype=np.float32)])
        stds = np.concatenate([np.std(seq_feats, axis=0), np.array([f0_std], dtype=np.float32)])
        pooled = np.concatenate([means, stds]).astype(np.float32)
        return _l2_normalize(pooled), active_s

    def embed(
        self,
        pcm16_or_f32: bytes | bytearray | memoryview | np.ndarray | Sequence[float],
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> list[float]:
        vec, _ = self.embed_with_duration(pcm16_or_f32, sample_rate)
        return vec


class OnnxSpeakerEmbedder:
    """Optional CPU-only ONNX neural speaker embedder (e.g., WeSpeaker/CAM++/ECAPA-TDNN)."""

    name = "onnx"

    def __init__(self, model_path: Path | str, *, intra_threads: int = 2) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX speaker model not found: {self.model_path}")
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        out_shape = self._session.get_outputs()[0].shape
        self.dim = int(out_shape[-1]) if out_shape and isinstance(out_shape[-1], int) else 192

    def embed_with_duration(
        self,
        pcm16_or_f32: bytes | bytearray | memoryview | np.ndarray | Sequence[float],
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> tuple[list[float], float]:
        x = _to_mono_f32(pcm16_or_f32, sample_rate)
        _, mask = _active_frames(x)
        active_s = float(np.sum(mask)) * (HOP_LEN / TARGET_SAMPLE_RATE)
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - 0.97 * x[:-1]
        y_frames = np.lib.stride_tricks.sliding_window_view(y, WIN_LEN)[::HOP_LEN] * _HAMMING
        spec = np.fft.rfft(y_frames, n=N_FFT, axis=1)
        power = (np.real(spec) ** 2 + np.imag(spec) ** 2).astype(np.float32)
        fbank = np.log(np.maximum(power[mask] @ _MEL_FB_80.T, 1e-10)).astype(np.float32)
        fbank -= np.mean(fbank, axis=0, keepdims=True)
        outputs = self._session.run(None, {self._input_name: fbank[None, :, :]})
        raw = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        self.dim = int(len(raw))
        return _l2_normalize(raw), active_s

    def embed(
        self,
        pcm16_or_f32: bytes | bytearray | memoryview | np.ndarray | Sequence[float],
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> list[float]:
        vec, _ = self.embed_with_duration(pcm16_or_f32, sample_rate)
        return vec


def make_speaker_embedder(
    backend: str = "auto",
    *,
    onnx_model_path: Path | str = "data/models/speaker.onnx",
) -> SpeakerEmbedder:
    kind = (backend or "auto").strip().lower()
    if kind == "acoustic":
        return AcousticEmbedder()
    if kind == "onnx":
        return OnnxSpeakerEmbedder(onnx_model_path)
    if kind == "auto":
        model_p = Path(onnx_model_path)
        if importlib.util.find_spec("onnxruntime") is not None and model_p.exists():
            try:
                return OnnxSpeakerEmbedder(model_p)
            except Exception:
                pass
        return AcousticEmbedder()
    raise ValueError(f"speaker backend must be 'auto', 'acoustic', or 'onnx', got {backend!r}")


class SpeakerRegistry:
    """Thread-safe atomic JSON persistence and cosine speaker matching."""

    def __init__(self, store_path: Path | str, *, embedder: SpeakerEmbedder | None = None) -> None:
        self.store_path = Path(store_path)
        self.embedder: SpeakerEmbedder = embedder or AcousticEmbedder()
        self._lock = threading.Lock()
        self.profiles: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        with self._lock:
            if not self.store_path.exists():
                self.profiles = {}
                return
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                raw_profiles = data.get("profiles") or {}
                loaded: dict[str, dict] = {}
                for role, entry in raw_profiles.items():
                    if isinstance(role, str) and isinstance(entry, dict) and isinstance(entry.get("embedding"), list):
                        loaded[role] = {
                            "embedding": [float(v) for v in entry["embedding"]],
                            "enrolled_at": float(entry.get("enrolled_at", 0.0)),
                            "duration_s": float(entry.get("duration_s", 0.0)),
                        }
                self.profiles = loaded
            except Exception:
                self.profiles = {}

    def _save_locked(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "backend": self.embedder.name,
            "profiles": self.profiles,
        }
        write_atomic(self.store_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def enrolled_roles(self) -> list[str]:
        with self._lock:
            return list(self.profiles.keys())

    def enroll(
        self,
        role: str,
        audio: bytes | bytearray | memoryview | np.ndarray | Sequence[float],
        sample_rate: int = TARGET_SAMPLE_RATE,
        *,
        players: Sequence[str] = (),
    ) -> dict:
        clean_role = (role or "").strip()
        if not clean_role:
            raise ValueError("speaker role cannot be empty")
        blocked = {p.strip().lower() for p in players if p and p.strip()}
        if clean_role.lower() in blocked:
            raise ValueError("use character or role names only, never real player names")

        if hasattr(self.embedder, "embed_with_duration"):
            emb, dur = self.embedder.embed_with_duration(audio, sample_rate)
        else:
            emb = self.embedder.embed(audio, sample_rate)
            dur = 0.0

        entry = {
            "embedding": [float(v) for v in emb],
            "enrolled_at": time.time(),
            "duration_s": round(float(dur), 3),
        }
        with self._lock:
            # Replace case-insensitive duplicate if someone re-enrolls "dm" vs "DM"
            existing_key = next((k for k in self.profiles if k.lower() == clean_role.lower()), None)
            if existing_key is not None and existing_key != clean_role:
                del self.profiles[existing_key]
            self.profiles[clean_role] = entry
            self._save_locked()
        return entry

    def remove(self, role: str) -> bool:
        target = (role or "").strip()
        if not target:
            return False
        with self._lock:
            key = target if target in self.profiles else next(
                (k for k in self.profiles if k.lower() == target.lower()), None
            )
            if key is None:
                return False
            del self.profiles[key]
            self._save_locked()
            return True

    def clear(self) -> None:
        with self._lock:
            self.profiles.clear()
            self._save_locked()

    def match(
        self,
        audio: bytes | bytearray | memoryview | np.ndarray | Sequence[float],
        sample_rate: int = TARGET_SAMPLE_RATE,
        *,
        threshold: float = 0.72,
        min_margin: float = 0.03,
    ) -> tuple[str | None, float]:
        with self._lock:
            snapshot = {k: list(v["embedding"]) for k, v in self.profiles.items()}
        if not snapshot:
            return None, 0.0
        try:
            emb = self.embedder.embed(audio, sample_rate)
        except InsufficientSpeechError:
            return None, 0.0
        scores = [
            (role, cosine_similarity(emb, ref))
            for role, ref in snapshot.items()
            if len(ref) == len(emb)
        ]
        if not scores:
            return None, 0.0
        scores.sort(key=lambda item: item[1], reverse=True)
        best_role, best_score = scores[0]
        if best_score < threshold:
            return None, float(best_score)
        if len(scores) >= 2 and (best_score - scores[1][1]) < min_margin:
            return None, float(best_score)
        return best_role, float(best_score)
