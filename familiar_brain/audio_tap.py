"""Transparent pure-stdlib asyncio HTTP/WS proxy with CPU-only speaker recognition tap.

Listens on `--listen-host:--listen-port` (default 0.0.0.0:8765) in front of
speech-to-speech on `--upstream-host:--upstream-port` (default 127.0.0.1:8766),
exposing loopback control endpoints on `--control-host:--control-port` (127.0.0.1:8767).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hashlib
import json
import logging
import math
import os
import struct
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from familiar_brain.ambient_audio import AmbientAudioAnalyzer
from familiar_brain.config import Speakers, load_config
from familiar_brain.speaker_id import (
    TARGET_SAMPLE_RATE,
    SpeakerRegistry,
    make_speaker_embedder,
)

log = logging.getLogger("familiar.audio_tap")

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_RING_SAMPLES = TARGET_SAMPLE_RATE * 10  # 10s rolling cap
_MAX_UTT_SAMPLES = TARGET_SAMPLE_RATE * 6    # 6s active utterance cap


def _ws_accept(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key.strip() + _WS_GUID).encode("ascii")).digest()).decode("ascii")


def _build_ws_frame(opcode: int, payload: bytes, *, mask: bool = False, fin: bool = True) -> bytes:
    b0 = (0x80 if fin else 0x00) | (opcode & 0x0F)
    length = len(payload)
    mask_bit = 0x80 if mask else 0x00
    if length < 126:
        header = struct.pack("!BB", b0, mask_bit | length)
    elif length < 65536:
        header = struct.pack("!BBH", b0, mask_bit | 126, length)
    else:
        header = struct.pack("!BBQ", b0, mask_bit | 127, length)
    if not mask:
        return header + payload
    key = os.urandom(4)
    if length:
        arr = np.frombuffer(payload, dtype=np.uint8)
        key_arr = np.frombuffer(key, dtype=np.uint8)
        masked = (arr ^ key_arr[np.arange(length) & 3]).tobytes()
    else:
        masked = b""
    return header + key + masked


async def _read_exact(reader: asyncio.StreamReader, n: int, pre_buf: bytearray) -> bytes:
    if n == 0:
        return b""
    if len(pre_buf) >= n:
        out = bytes(pre_buf[:n])
        del pre_buf[:n]
        return out
    if pre_buf:
        head = bytes(pre_buf)
        pre_buf.clear()
        return head + await reader.readexactly(n - len(head))
    return await reader.readexactly(n)


async def _read_ws_frame(reader: asyncio.StreamReader, pre_buf: bytearray) -> tuple[bool, int, bytes]:
    hdr = await _read_exact(reader, 2, pre_buf)
    b0, b1 = hdr[0], hdr[1]
    fin = bool(b0 & 0x80)
    opcode = b0 & 0x0F
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F
    if length == 126:
        (length,) = struct.unpack("!H", await _read_exact(reader, 2, pre_buf))
    elif length == 127:
        (length,) = struct.unpack("!Q", await _read_exact(reader, 8, pre_buf))
    mask_key = await _read_exact(reader, 4, pre_buf) if masked else b""
    payload = await _read_exact(reader, length, pre_buf) if length else b""
    if masked and payload:
        arr = np.frombuffer(payload, dtype=np.uint8)
        key_arr = np.frombuffer(mask_key, dtype=np.uint8)
        payload = (arr ^ key_arr[np.arange(len(payload)) & 3]).tobytes()
    return fin, opcode, payload


async def _read_http_head(reader: asyncio.StreamReader) -> tuple[str, str, str, dict[str, str], list[tuple[str, str]], bytes] | None:
    buf = bytearray()
    while b"\r\n\r\n" not in buf:
        chunk = await reader.read(8192)
        if not chunk:
            return None
        buf.extend(chunk)
        if len(buf) > 1024 * 1024:
            raise ValueError("HTTP header too large")
    idx = buf.index(b"\r\n\r\n") + 4
    head_bytes = bytes(buf[:idx])
    remainder = bytes(buf[idx:])
    lines = head_bytes.decode("latin-1").split("\r\n")
    req_parts = lines[0].split(" ", 2)
    if len(req_parts) < 2:
        raise ValueError("malformed HTTP request line")
    method = req_parts[0].upper()
    target = req_parts[1]
    version = req_parts[2] if len(req_parts) > 2 else "HTTP/1.1"
    raw_headers: list[tuple[str, str]] = []
    headers_lc: dict[str, str] = {}
    for ln in lines[1:]:
        if not ln:
            continue
        k, sep, v = ln.partition(":")
        if sep:
            k_s, v_s = k.strip(), v.strip()
            raw_headers.append((k_s, v_s))
            headers_lc[k_s.lower()] = v_s
    return method, target, version, headers_lc, raw_headers, remainder


def _decode_pcm16_b64(raw_b64: str, src_rate: int) -> np.ndarray | None:
    if not raw_b64:
        return None
    try:
        raw = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError):
        return None
    usable = len(raw) - (len(raw) % 2)
    if usable <= 0:
        return None
    arr = np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32) / 32768.0
    if src_rate != TARGET_SAMPLE_RATE and src_rate > 0 and len(arr) > 0:
        g = math.gcd(TARGET_SAMPLE_RATE, int(src_rate))
        arr = resample_poly(arr, TARGET_SAMPLE_RATE // g, int(src_rate) // g).astype(np.float32, copy=False)
    return arr


def _f32_to_pcm16_b64(samples: np.ndarray) -> str:
    if len(samples) == 0:
        return ""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


class AudioTapProxy:
    """Asyncio HTTP + RFC 6455 WebSocket proxy with non-blocking CPU speaker-ID tap."""

    def __init__(
        self,
        *,
        listen_host: str = "0.0.0.0",
        listen_port: int = 8765,
        upstream_host: str = "127.0.0.1",
        upstream_port: int = 8766,
        control_host: str = "127.0.0.1",
        control_port: int = 8767,
        speakers_cfg: Speakers | None = None,
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.control_host = control_host
        self.control_port = control_port
        self.cfg = speakers_cfg or Speakers(enabled=True)
        embedder = make_speaker_embedder(
            self.cfg.backend, onnx_model_path=self.cfg.onnx_model_path
        )
        self.registry = SpeakerRegistry(self.cfg.store_path, embedder=embedder)

        self._audio_queue: deque[tuple[str, Any, int]] = deque(maxlen=400)
        self._wake_event: asyncio.Event | None = None
        self._ring_chunks: deque[np.ndarray] = deque()
        self._ring_samples = 0
        self._utt_chunks: deque[np.ndarray] = deque()
        self._utt_samples = 0
        self._samples_since_match = 0
        self._last_match_at = 0.0
        self._stream_misses = 0

        self._capturing = False
        self._capture_chunks: list[np.ndarray] = []
        self._capture_lock: asyncio.Lock | None = None

        self._current: dict[str, Any] = {
            "role": None,
            "confidence": None,
            "updated_at": None,
        }
        self.ambient = AmbientAudioAnalyzer(sample_rate=TARGET_SAMPLE_RATE)
        self._last_ambient_at = 0.0
        self._reload_seq = 0
        self._proxy_server: asyncio.Server | None = None
        self._control_server: asyncio.Server | None = None
        self._worker_task: asyncio.Task[None] | None = None

    def current_state(self) -> dict[str, Any]:
        return {
            "role": self._current["role"],
            "confidence": self._current["confidence"],
            "updated_at": self._current["updated_at"],
            "capturing": bool(self._capturing),
        }

    def _append_ring(self, arr: np.ndarray) -> None:
        self._ring_chunks.append(arr)
        self._ring_samples += len(arr)
        while self._ring_samples > _MAX_RING_SAMPLES and self._ring_chunks:
            dropped = self._ring_chunks.popleft()
            self._ring_samples -= len(dropped)

    def _append_utt(self, arr: np.ndarray) -> None:
        self._utt_chunks.append(arr)
        self._utt_samples += len(arr)
        self._samples_since_match += len(arr)
        while self._utt_samples > _MAX_UTT_SAMPLES and self._utt_chunks:
            dropped = self._utt_chunks.popleft()
            self._utt_samples -= len(dropped)

    def _clear_utt(self) -> None:
        self._utt_chunks.clear()
        self._utt_samples = 0
        self._samples_since_match = 0

    def _enqueue_tap(self, kind: str, payload: Any, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
        if not self._capturing and not self.registry.profiles:
            if kind != "audio":
                return
            now = time.monotonic()
            if now - self._last_ambient_at < 0.25:
                return
            self._last_ambient_at = now
        self._audio_queue.append((kind, payload, sample_rate))
        if self._wake_event is not None:
            self._wake_event.set()

    async def _run_match(self, audio: np.ndarray, *, clear_on_miss: bool, misses_to_clear: int = 1) -> None:
        if len(audio) < 400 or not self.registry.profiles:
            if clear_on_miss:
                self._current = {"role": None, "confidence": None, "updated_at": time.time()}
            return
        seq = self._reload_seq
        role, conf = await asyncio.to_thread(
            self.registry.match,
            audio,
            TARGET_SAMPLE_RATE,
            threshold=self.cfg.threshold,
            min_margin=self.cfg.min_margin,
        )
        if seq != self._reload_seq:
            return
        now = time.time()
        if role is not None and role in self.registry.profiles:
            self._stream_misses = 0
            self._current = {
                "role": role,
                "confidence": round(float(conf), 4),
                "updated_at": now,
            }
        elif clear_on_miss:
            if misses_to_clear > 1:
                # Streaming: a silent window (conf 0) leaves expiry to max_age_s, and one noisy
                # window (a laugh, dice) must not drop the DM mid-turn; clear on repeated misses.
                if conf <= 0:
                    return
                self._stream_misses += 1
                if self._stream_misses < misses_to_clear:
                    return
            self._current = {
                "role": None,
                "confidence": round(float(conf), 4) if conf > 0 else None,
                "updated_at": now,
            }

    async def _tap_worker(self) -> None:
        assert self._wake_event is not None
        while True:
            await self._wake_event.wait()
            self._wake_event.clear()
            try:
                await self._drain_tap_queue()
            except Exception:
                # One bad chunk or embedder error used to kill speaker ID for the rest of the night.
                log.warning("tap worker error; dropping this batch and continuing", exc_info=True)
                await asyncio.sleep(0.1)

    async def _drain_tap_queue(self) -> None:
        assert self._wake_event is not None
        commit_requested = False
        while self._audio_queue:
            kind, payload, sr = self._audio_queue.popleft()
            if kind == "commit":
                commit_requested = True
                break
            if kind == "audio" and isinstance(payload, str):
                arr = _decode_pcm16_b64(payload, sr)
                if arr is None or len(arr) == 0:
                    continue
                self._append_ring(arr)
                self._append_utt(arr)
                self.ambient.feed(arr)
                if self._capturing:
                    self._capture_chunks.append(arr)
        if commit_requested:
            if self._utt_chunks:
                buf = np.concatenate(list(self._utt_chunks))
                self._clear_utt()
                await self._run_match(buf, clear_on_miss=True)
                self._last_match_at = time.monotonic()
            else:
                self._current = {"role": None, "confidence": None, "updated_at": time.time()}
            if self._audio_queue:
                self._wake_event.set()
        elif (
            self.registry.profiles
            and self._utt_samples >= TARGET_SAMPLE_RATE // 2
            and self._samples_since_match >= TARGET_SAMPLE_RATE // 2
            and (time.monotonic() - self._last_match_at) >= 0.35
        ):
            buf = np.concatenate(list(self._utt_chunks))
            if len(buf) > TARGET_SAMPLE_RATE * 3:
                buf = buf[-(TARGET_SAMPLE_RATE * 3):]
            self._samples_since_match = 0
            self._last_match_at = time.monotonic()
            # Clients never send commit, so repeated misses must clear the label or the DM sticks.
            await self._run_match(buf, clear_on_miss=True, misses_to_clear=2)

    def _inspect_client_text(self, payload: bytes, state: dict[str, int]) -> None:
        if b'"input_audio_buffer.' not in payload and b'"session.update"' not in payload:
            return
        try:
            msg = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(msg, dict):
            return
        mtype = msg.get("type")
        if mtype == "session.update":
            sess = msg.get("session") or {}
            if isinstance(sess, dict):
                sr = sess.get("input_audio_sample_rate")
                if not isinstance(sr, int):
                    # GA realtime shape: session.audio.input.format.rate
                    node: Any = sess
                    for key in ("audio", "input", "format", "rate"):
                        node = node.get(key) if isinstance(node, dict) else None
                    sr = node
                if isinstance(sr, int) and sr > 0:
                    state["sr"] = sr
                elif sess.get("input_audio_format") == "pcm24":
                    state["sr"] = 24000
        elif mtype == "input_audio_buffer.append":
            raw_b64 = msg.get("audio")
            if isinstance(raw_b64, str) and raw_b64:
                self._enqueue_tap("audio", raw_b64, state["sr"])
        elif mtype == "input_audio_buffer.commit":
            self._enqueue_tap("commit", None, state["sr"])

    async def _handle_ws(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        target: str,
        headers_lc: dict[str, str],
        raw_headers: list[tuple[str, str]],
        remainder: bytes,
    ) -> None:
        ws_key = headers_lc.get("sec-websocket-key", "")
        if not ws_key:
            client_writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await client_writer.drain()
            return
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(self.upstream_host, self.upstream_port),
                timeout=1.5,
            )
        except (OSError, TimeoutError):
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await client_writer.drain()
            return

        up_key = base64.b64encode(os.urandom(16)).decode("ascii")
        req_lines = [
            f"GET {target} HTTP/1.1",
            f"Host: {self.upstream_host}:{self.upstream_port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {up_key}",
            f"Sec-WebSocket-Version: {headers_lc.get('sec-websocket-version', '13')}",
        ]
        skip = {
            "host", "upgrade", "connection", "sec-websocket-key",
            "sec-websocket-version", "sec-websocket-extensions",
        }
        for k, v in raw_headers:
            if k.lower() not in skip:
                req_lines.append(f"{k}: {v}")
        up_writer.write(("\r\n".join(req_lines) + "\r\n\r\n").encode("latin-1"))
        await up_writer.drain()

        try:
            up_buf = bytearray()
            while b"\r\n\r\n" not in up_buf:
                chunk = await asyncio.wait_for(up_reader.read(4096), timeout=3.0)
                if not chunk:
                    raise ConnectionError("upstream closed during WS handshake")
                up_buf.extend(chunk)
            sep_idx = up_buf.index(b"\r\n\r\n") + 4
            status_line = bytes(up_buf[:sep_idx]).decode("latin-1").split("\r\n", 1)[0]
            if "101" not in status_line:
                client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                await client_writer.drain()
                up_writer.close()
                return
        except Exception:
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await client_writer.drain()
            up_writer.close()
            return

        client_pre = bytearray(remainder)
        up_pre = bytearray(up_buf[sep_idx:])

        accept = _ws_accept(ws_key)
        client_writer.write(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode("latin-1")
        )
        await client_writer.drain()

        sess_state = {"sr": TARGET_SAMPLE_RATE}

        async def pump_client_to_upstream() -> None:
            frag_buf = bytearray()
            frag_is_text = False
            while True:
                fin, op, payload = await _read_ws_frame(client_reader, client_pre)
                up_writer.write(_build_ws_frame(op, payload, mask=True, fin=fin))
                await up_writer.drain()
                if op == 0x8:
                    break
                if op == 0x1:
                    if fin and not frag_buf:
                        self._inspect_client_text(payload, sess_state)
                    else:
                        frag_is_text = True
                        if len(frag_buf) + len(payload) <= 2 * 1024 * 1024:
                            frag_buf.extend(payload)
                        if fin:
                            self._inspect_client_text(bytes(frag_buf), sess_state)
                            frag_buf.clear()
                            frag_is_text = False
                elif op == 0x0 and frag_is_text:
                    if len(frag_buf) + len(payload) <= 2 * 1024 * 1024:
                        frag_buf.extend(payload)
                    if fin:
                        self._inspect_client_text(bytes(frag_buf), sess_state)
                        frag_buf.clear()
                        frag_is_text = False

        async def pump_upstream_to_client() -> None:
            while True:
                fin, op, payload = await _read_ws_frame(up_reader, up_pre)
                client_writer.write(_build_ws_frame(op, payload, mask=False, fin=fin))
                await client_writer.drain()
                if op == 0x8:
                    break

        t_c2u = asyncio.create_task(pump_client_to_upstream())
        t_u2c = asyncio.create_task(pump_upstream_to_client())
        try:
            done, pending = await asyncio.wait({t_c2u, t_u2c}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                try:
                    t.result()
                except (asyncio.IncompleteReadError, ConnectionError, OSError, asyncio.CancelledError):
                    pass
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            for w in (up_writer, client_writer):
                try:
                    w.close()
                except Exception:
                    pass

    async def _handle_http_proxy(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        method: str,
        target: str,
        headers_lc: dict[str, str],
        raw_headers: list[tuple[str, str]],
        remainder: bytes,
    ) -> None:
        clen = int(headers_lc.get("content-length") or 0)
        body = bytearray(remainder)
        if len(body) < clen:
            body.extend(await client_reader.readexactly(clen - len(body)))
        req_body = bytes(body[:clen])

        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(self.upstream_host, self.upstream_port),
                timeout=0.8,
            )
        except (OSError, TimeoutError):
            err = json.dumps({"error": "upstream speech-to-speech unavailable"}).encode("utf-8")
            client_writer.write(
                (
                    "HTTP/1.1 502 Bad Gateway\r\n"
                    "Content-Type: application/json; charset=utf-8\r\n"
                    f"Content-Length: {len(err)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("latin-1")
                + err
            )
            await client_writer.drain()
            return

        try:
            lines = [f"{method} {target} HTTP/1.1", f"Host: {self.upstream_host}:{self.upstream_port}"]
            for k, v in raw_headers:
                if k.lower() not in ("host", "connection"):
                    lines.append(f"{k}: {v}")
            lines.append("Connection: close")
            up_writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + req_body)
            await up_writer.drain()

            while True:
                chunk = await asyncio.wait_for(up_reader.read(65536), timeout=5.0)
                if not chunk:
                    break
                client_writer.write(chunk)
                await client_writer.drain()
        except Exception:
            pass
        finally:
            try:
                up_writer.close()
            except Exception:
                pass

    async def _handle_public_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            parsed = await _read_http_head(reader)
            if parsed is None:
                return
            method, target, _ver, headers_lc, raw_headers, remainder = parsed
            if headers_lc.get("upgrade", "").lower() == "websocket":
                await self._handle_ws(reader, writer, target, headers_lc, raw_headers, remainder)
            else:
                await self._handle_http_proxy(reader, writer, method, target, headers_lc, raw_headers, remainder)
        except Exception as exc:
            log.debug("public connection closed: %s", exc)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _send_json_response(self, writer: asyncio.StreamWriter, status: int, reason: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        writer.write(
            (
                f"HTTP/1.1 {status} {reason}\r\n"
                "Content-Type: application/json; charset=utf-8\r\n"
                f"Content-Length: {len(data)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("latin-1")
            + data
        )
        await writer.drain()

    async def _handle_control_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            parsed = await _read_http_head(reader)
            if parsed is None:
                return
            method, target, _ver, headers_lc, _raw_headers, remainder = parsed
            path = target.split("?", 1)[0]
            clen = int(headers_lc.get("content-length") or 0)
            body = bytearray(remainder)
            if len(body) < clen:
                body.extend(await reader.readexactly(clen - len(body)))
            raw_body = bytes(body[:clen])

            if method == "GET" and path == "/_tap/current":
                await self._send_json_response(writer, 200, "OK", self.current_state())
                return

            if method == "GET" and path == "/_tap/ambient":
                await self._send_json_response(writer, 200, "OK", self.ambient.snapshot().to_dict())
                return

            if method == "POST" and path == "/_tap/reload":
                self._reload_seq += 1
                try:
                    req_data = json.loads(raw_body.decode("utf-8") or "{}")
                except Exception:
                    req_data = {}
                if isinstance(req_data, dict) and isinstance(req_data.get("store_path"), str) and req_data["store_path"].strip():
                    self.registry.store_path = Path(req_data["store_path"].strip())
                self.registry.load()
                if not self.registry.profiles or (
                    self._current["role"] is not None
                    and self._current["role"] not in self.registry.profiles
                ):
                    self._current = {"role": None, "confidence": None, "updated_at": None}
                await self._send_json_response(
                    writer, 200, "OK", {"ok": True, "enrolled": self.registry.enrolled_roles()}
                )
                return

            if method == "POST" and path == "/_tap/capture":
                assert self._capture_lock is not None
                if self._capture_lock.locked():
                    await self._send_json_response(writer, 409, "Conflict", {"error": "capture already in progress"})
                    return
                try:
                    req_data = json.loads(raw_body.decode("utf-8") or "{}")
                except Exception:
                    req_data = {}
                dur = req_data.get("duration_s", 5.0) if isinstance(req_data, dict) else 5.0
                try:
                    duration_s = max(0.2, min(30.0, float(dur)))
                except (TypeError, ValueError):
                    duration_s = 5.0

                async with self._capture_lock:
                    self._capture_chunks = []
                    self._capturing = True
                    try:
                        await asyncio.sleep(duration_s)
                        if self._audio_queue and self._wake_event is not None:
                            self._wake_event.set()
                            await asyncio.sleep(0.05)
                    finally:
                        self._capturing = False
                    chunks = list(self._capture_chunks)
                    self._capture_chunks = []
                if not chunks:
                    await self._send_json_response(writer, 408, "Request Timeout", {"error": "no audio captured"})
                    return
                pcm = np.concatenate(chunks)
                b64 = _f32_to_pcm16_b64(pcm)
                await self._send_json_response(
                    writer,
                    200,
                    "OK",
                    {
                        "audio_b64": b64,
                        "sample_rate": TARGET_SAMPLE_RATE,
                        "duration_s": round(len(pcm) / TARGET_SAMPLE_RATE, 3),
                    },
                )
                return

            await self._send_json_response(writer, 404, "Not Found", {"error": "not found"})
        except Exception as exc:
            log.debug("control connection error: %s", exc)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def start(self) -> None:
        self._wake_event = asyncio.Event()
        self._capture_lock = asyncio.Lock()
        self._worker_task = asyncio.create_task(self._tap_worker())
        self._proxy_server = await asyncio.start_server(
            self._handle_public_conn, self.listen_host, self.listen_port
        )
        self._control_server = await asyncio.start_server(
            self._handle_control_conn, self.control_host, self.control_port
        )
        p_sock = self._proxy_server.sockets[0].getsockname()
        c_sock = self._control_server.sockets[0].getsockname()
        self.listen_port = int(p_sock[1])
        self.control_port = int(c_sock[1])
        log.info(
            "audio_tap listening on %s:%d -> %s:%d (control %s:%d)",
            self.listen_host, self.listen_port,
            self.upstream_host, self.upstream_port,
            self.control_host, self.control_port,
        )

    async def serve_forever(self) -> None:
        await self.start()
        assert self._proxy_server is not None and self._control_server is not None
        try:
            await asyncio.gather(
                self._proxy_server.serve_forever(),
                self._control_server.serve_forever(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            await self.close()

    async def close(self) -> None:
        for srv in (self._proxy_server, self._control_server):
            if srv is not None:
                srv.close()
        if self._worker_task is not None:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
            self._worker_task = None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Opt-in CPU-only speaker recognition WS/HTTP proxy")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8765)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=8766)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8767)
    parser.add_argument("--config", default="config/brain.toml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg_path = Path(args.config)
    speakers_cfg = load_config(cfg_path).speakers if cfg_path.exists() else Speakers(enabled=True)
    proxy = AudioTapProxy(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream_host=args.upstream_host,
        upstream_port=args.upstream_port,
        control_host=args.control_host,
        control_port=args.control_port,
        speakers_cfg=speakers_cfg,
    )
    try:
        asyncio.run(proxy.serve_forever())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
