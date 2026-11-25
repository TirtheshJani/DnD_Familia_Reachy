"""Pure-stdlib WebSocket bridge between Reachy Mini media/tools and laptop :8765/v1/realtime."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import socket
import struct
import threading
import time
from typing import Any, Callable

import numpy as np

from reachy_familiar.creature import CreatureController

log = logging.getLogger("reachy_familiar.audio_bridge")

REALTIME_SAMPLE_RATE = 24000
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _resample_f32(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample 1-D float32 audio between hardware rate (e.g. 16 kHz) and Realtime 24 kHz."""
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0 or src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return arr
    try:
        from scipy.signal import resample_poly  # type: ignore[import-not-found]

        g = math.gcd(int(src_rate), int(dst_rate))
        return resample_poly(arr, int(dst_rate) // g, int(src_rate) // g).astype(np.float32, copy=False)
    except Exception:
        dst_len = max(1, int(round(arr.size * (float(dst_rate) / float(src_rate)))))
        x_old = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        return np.interp(x_new, x_old, arr).astype(np.float32)
DEFAULT_TOOLS = [
    {
        "type": "function",
        "name": "play_emotion",
        "description": "Play an expressive creature emotion on Reachy Mini.",
        "parameters": {"type": "object", "properties": {"emotion": {"type": "string"}}, "required": ["emotion"]},
    },
    {
        "type": "function",
        "name": "stop_emotion",
        "description": "Stop the active emotion.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "head_tracking",
        "description": "Enable or disable face tracking.",
        "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"]},
    },
    {
        "type": "function",
        "name": "move_head",
        "description": "Glance in a direction (left, right, up, down, front).",
        "parameters": {"type": "object", "properties": {"direction": {"type": "string"}}, "required": ["direction"]},
    },
    {
        "type": "function",
        "name": "idle_do_nothing",
        "description": "Stay silent when not addressed.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "camera",
        "description": "Capture a photo from Reachy Mini's camera to see the D&D table.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}},
    },
]


class SimpleWebSocket:
    """Minimal RFC 6455 client using only Python stdlib."""

    def __init__(self, host: str, port: int, path: str = "/v1/realtime", timeout: float = 10.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self._send_lock = threading.Lock()
        self._buf = bytearray()
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
        self.sock.sendall(req)
        while b"\r\n\r\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("WebSocket handshake closed by server")
            self._buf.extend(chunk)
        sep = self._buf.index(b"\r\n\r\n") + 4
        head = bytes(self._buf[:sep]).decode("latin-1")
        del self._buf[:sep]
        lines = head.split("\r\n")
        headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}
        expected = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()).decode("ascii")
        if lines[0].split()[1:2] != ["101"] or headers.get("sec-websocket-accept") != expected:
            self.sock.close()
            raise ConnectionError(f"WebSocket upgrade refused: {lines[0]}")

    def send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode("utf-8"))

    def send_json(self, obj: dict[str, Any]) -> None:
        self.send_text(json.dumps(obj, ensure_ascii=False))

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        if n < 126:
            head = struct.pack(">BB", 0x80 | opcode, 0x80 | n)
        elif n < 65536:
            head = struct.pack(">BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            head = struct.pack(">BBQ", 0x80 | opcode, 0x80 | 127, n)
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        with self._send_lock:
            self.sock.sendall(head + mask + masked)

    def _take(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("WebSocket closed")
            self._buf.extend(chunk)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def recv_text(self, timeout: float = 1.0) -> str | None:
        self.sock.settimeout(timeout)
        parts: list[bytes] = []
        while True:
            try:
                b0, b1 = self._take(2)
            except socket.timeout:
                return None
            n = b1 & 0x7F
            if n == 126:
                (n,) = struct.unpack(">H", self._take(2))
            elif n == 127:
                (n,) = struct.unpack(">Q", self._take(8))
            mask = self._take(4) if (b1 & 0x80) else b""
            payload = self._take(n)
            if mask:
                payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
            op = b0 & 0x0F
            if op == 0x8:
                raise ConnectionError("WebSocket close frame received")
            if op == 0x9:
                self._send_frame(0xA, payload)
                continue
            if op in (0x0, 0x1, 0x2):
                parts.append(payload)
                if b0 & 0x80:
                    return b"".join(parts).decode("utf-8", errors="replace")

    def close(self) -> None:
        try:
            self._send_frame(0x8, struct.pack(">H", 1000))
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


def _capture_camera_b64(reachy_mini: Any) -> str | None:
    """Capture one JPEG data-URI frame in memory (never written to disk)."""
    media = getattr(reachy_mini, "media", None)
    if media is None or not hasattr(media, "get_frame"):
        return None
    try:
        frame = media.get_frame()
        if frame is None:
            return None
        try:
            import cv2  # type: ignore[import-not-found]

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        except ImportError:
            pass
        from PIL import Image  # type: ignore[import-not-found]

        img = Image.fromarray(frame)
        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(bio.getvalue()).decode("ascii")
    except Exception as exc:
        log.warning("camera capture failed: %s", exc)
        return None


def apply_speaker_boost(samples: np.ndarray, boost_db: float) -> np.ndarray:
    """Apply speaker loudness gain (0..15 dB) with soft-clipping limiter."""
    arr = np.asarray(samples, dtype=np.float32)
    db = max(0.0, min(15.0, float(boost_db)))
    if arr.size == 0 or db <= 0.05:
        return np.clip(arr, -1.0, 1.0)
    gain = float(10.0 ** (db / 20.0))
    scaled = arr * gain
    # Smooth tanh limiter keeps loudness punchy without harsh digital clipping.
    return np.tanh(scaled).astype(np.float32, copy=False)


class RealtimeAudioBridge:
    """Streams Reachy Mini audio to laptop :8765/v1/realtime and executes robot tool calls."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        reachy_mini: Any,
        creature: CreatureController,
        on_connected: Callable[[bool], None] | None = None,
        speaker_boost_db: float = 6.0,
    ) -> None:
        self.host = host
        self.port = port
        self.reachy = reachy_mini
        self.creature = creature
        self.on_connected = on_connected
        self.speaking = False
        self.connected = False
        self.mic_level = 0.0
        self.speaker_boost_db = max(0.0, min(15.0, float(speaker_boost_db)))
        self._last_audio_out_at = 0.0
        self._barge_in_muted = False
        self._reconnect_req = threading.Event()
        self._ws: SimpleWebSocket | None = None

    def set_speaker_boost_db(self, boost_db: float) -> None:
        self.speaker_boost_db = max(0.0, min(15.0, float(boost_db)))

    def request_reconnect(self) -> None:
        """Close the active WebSocket so run_loop reconnects cleanly right away."""
        self._reconnect_req.set()
        ws = self._ws
        if ws is not None:
            ws.close()

    def play_test_tone(self) -> dict[str, Any]:
        """Play a crisp 3-note chime through Reachy's speaker using current boost_db."""
        media = getattr(self.reachy, "media", None)
        if media is None or not hasattr(media, "push_audio_sample"):
            return {"ok": False, "error": "Reachy speaker media unavailable"}
        try:
            get_out_sr = getattr(media, "get_output_audio_samplerate", None)
            sr = int(get_out_sr()) if callable(get_out_sr) else 16000
            if sr <= 0:
                sr = 16000
            notes = (523.25, 659.25, 783.99)  # C5 -> E5 -> G5 brass chime
            chunks = []
            for freq in notes:
                n = int(sr * 0.16)
                t = np.arange(n, dtype=np.float32) / float(sr)
                env = np.sin(np.pi * np.arange(n, dtype=np.float32) / float(max(1, n - 1)))
                chunks.append((0.28 * env * np.sin(2.0 * math.pi * freq * t)).astype(np.float32))
            tone = apply_speaker_boost(np.concatenate(chunks), self.speaker_boost_db)
            media.push_audio_sample(tone)
            self.creature.play_emotion("welcoming", force=True)
            return {
                "ok": True,
                "detail": f"Played speaker test chime (+{self.speaker_boost_db:.1f} dB) and welcoming nod.",
            }
        except Exception as exc:
            return {"ok": False, "error": f"speaker test failed: {exc}"}

    def run_loop(self, stop_event: threading.Event) -> None:
        """Connect, stream audio, handle tool calls, and auto-reconnect until stop_event."""
        media = getattr(self.reachy, "media", None)
        if media is not None:
            for starter in ("start_recording", "start_playing"):
                fn = getattr(media, starter, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as exc:
                        log.warning("%s failed: %s", starter, exc)

        try:
            while not stop_event.is_set():
                fast_retry = False
                try:
                    self._reconnect_req.clear()
                    self._run_session(stop_event)
                except Exception as exc:
                    fast_retry = self._reconnect_req.is_set()
                    if fast_retry:
                        log.info("realtime WS reconnecting on user request")
                    else:
                        log.warning("realtime WS disconnected (%s); retrying in 2s", exc)
                finally:
                    self.connected = False
                    self.speaking = False
                    self.creature.set_wobbling(False)
                    if self.on_connected:
                        self.on_connected(False)
                    if self._ws is not None:
                        self._ws.close()
                        self._ws = None
                stop_event.wait(0.25 if fast_retry else 2.0)
        finally:
            if media is not None:
                for stopper in ("stop_recording", "stop_playing"):
                    fn = getattr(media, stopper, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass

    def _run_session(self, stop_event: threading.Event) -> None:
        ws = SimpleWebSocket(self.host, self.port, "/v1/realtime", timeout=10.0)
        self._ws = ws
        self.connected = True
        self._barge_in_muted = False
        if self.on_connected:
            self.on_connected(True)

        # OpenAI Realtime GA session.update schema (validated by speech-to-speech 1.0+).
        ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": (
                        "You are Rocky, a small familiar at a game of Dungeons and Dragons. "
                        "Follow the fuller instructions you are given with each turn."
                    ),
                    "audio": {
                        "input": {"format": {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE}},
                        "output": {"format": {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE}},
                    },
                    "tools": DEFAULT_TOOLS,
                },
            }
        )

        mic_stop = threading.Event()
        mic_thread = threading.Thread(
            target=self._mic_loop,
            args=(ws, stop_event, mic_stop),
            daemon=True,
            name="reachy-mic-tx",
        )
        mic_thread.start()
        try:
            while not stop_event.is_set():
                raw = ws.recv_text(timeout=0.5)
                if self.speaking and (time.monotonic() - self._last_audio_out_at) > 0.6:
                    self.speaking = False
                    self.creature.set_wobbling(False)
                if raw is None:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self._handle_server_event(ws, ev)
        finally:
            mic_stop.set()
            mic_thread.join(timeout=2.0)

    def _mic_loop(
        self, ws: SimpleWebSocket, stop_event: threading.Event, mic_stop: threading.Event
    ) -> None:
        media = getattr(self.reachy, "media", None)
        if media is None or not hasattr(media, "get_audio_sample"):
            return
        get_in_sr = getattr(media, "get_input_audio_samplerate", None)
        in_sr = int(get_in_sr()) if callable(get_in_sr) else REALTIME_SAMPLE_RATE
        if in_sr <= 0:
            in_sr = 16000
        while not stop_event.is_set() and not mic_stop.is_set():
            try:
                sample = media.get_audio_sample()
                if sample is None:
                    time.sleep(0.02)
                    continue
                arr = np.asarray(sample)
                if arr.size == 0:
                    time.sleep(0.02)
                    continue
                if arr.ndim == 2:
                    arr = arr.mean(axis=1)
                if np.issubdtype(arr.dtype, np.floating):
                    f32 = arr.astype(np.float32, copy=False)
                else:
                    f32 = arr.astype(np.float32) / 32768.0
                # Track DC-removed RMS mic level for the Companion UI meter.
                centered = f32 - float(np.mean(f32))
                rms = float(np.sqrt(np.mean(centered * centered) + 1e-12))
                lvl = min(1.0, rms * 5.0)
                self.mic_level = round(0.65 * self.mic_level + 0.35 * lvl, 3)
                if in_sr != REALTIME_SAMPLE_RATE:
                    f32 = _resample_f32(f32, in_sr, REALTIME_SAMPLE_RATE)
                pcm16 = np.clip(f32 * 32767.0, -32768, 32767).astype("<i2")
                ws.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm16.tobytes()).decode("ascii"),
                    }
                )
            except Exception as exc:
                log.debug("mic loop ending: %s", exc)
                break

    def _handle_server_event(self, ws: SimpleWebSocket, ev: dict[str, Any]) -> None:
        etype = ev.get("type")
        if etype == "input_audio_buffer.speech_started":
            # Full barge-in: drop any remaining output chunks from the interrupted reply.
            self._barge_in_muted = True
            if self.speaking:
                self.speaking = False
                self.creature.set_wobbling(False)
            media = getattr(self.reachy, "media", None)
            audio = getattr(media, "audio", None) if media is not None else None
            for obj, meth in ((media, "clear_audio_buffer"), (audio, "clear_output_buffer")):
                fn = getattr(obj, meth, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
            return
        if etype == "response.created":
            self._barge_in_muted = False
            return
        if etype in ("response.audio.delta", "response.output_audio.delta"):
            if self._barge_in_muted:
                return
            b64 = ev.get("delta") or ""
            if b64:
                self._play_audio_b64(b64)
            return
        if etype in ("response.audio.done", "response.output_audio.done", "response.done"):
            self._barge_in_muted = False
            if self.speaking:
                self.speaking = False
                self.creature.set_wobbling(False)
            return
        if etype == "response.function_call_arguments.done":
            self._dispatch_tool(
                ws,
                name=str(ev.get("name") or ""),
                call_id=str(ev.get("call_id") or ""),
                raw_args=str(ev.get("arguments") or "{}"),
            )

    def _play_audio_b64(self, b64: str) -> None:
        media = getattr(self.reachy, "media", None)
        if media is None or not hasattr(media, "push_audio_sample"):
            return
        try:
            pcm16 = np.frombuffer(base64.b64decode(b64), dtype="<i2")
            f32 = pcm16.astype(np.float32) / 32768.0
            get_out_sr = getattr(media, "get_output_audio_samplerate", None)
            out_sr = int(get_out_sr()) if callable(get_out_sr) else REALTIME_SAMPLE_RATE
            if out_sr <= 0:
                out_sr = 16000
            if out_sr != REALTIME_SAMPLE_RATE:
                f32 = _resample_f32(f32, REALTIME_SAMPLE_RATE, out_sr)
            f32 = apply_speaker_boost(f32, self.speaker_boost_db)
            if not self.speaking:
                self.speaking = True
                self.creature.set_wobbling(True)
            self._last_audio_out_at = time.monotonic()
            media.push_audio_sample(f32)
        except Exception as exc:
            log.debug("push_audio_sample failed: %s", exc)

    def _dispatch_tool(
        self, ws: SimpleWebSocket, *, name: str, call_id: str, raw_args: str
    ) -> None:
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}

        if name == "idle_do_nothing":
            result: dict[str, Any] = {"status": "idle"}
        elif name == "play_emotion":
            move = self.creature.play_emotion(str(args.get("emotion") or "attentive"))
            result = {"status": "played", "move": move}
        elif name == "stop_emotion":
            self.creature.stop_emotion()
            result = {"status": "stopped"}
        elif name == "head_tracking":
            enabled = bool(args.get("enabled", True))
            self.creature.set_tracking(enabled)
            result = {"status": "ok", "enabled": enabled}
        elif name == "move_head":
            direction = str(args.get("direction") or "front")
            self.creature.move_head(direction)
            result = {"status": "ok", "direction": direction}
        elif name == "camera":
            img_url = _capture_camera_b64(self.reachy)
            if img_url:
                ws.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps({"image_attached": True}),
                        },
                    }
                )
                ws.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": img_url}],
                        },
                    }
                )
                ws.send_json({"type": "response.create"})
                return
            result = {"error": "camera frame unavailable"}
        else:
            result = {"error": f"unknown tool {name}"}

        if call_id:
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    },
                }
            )
