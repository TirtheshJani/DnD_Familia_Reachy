"""Thin runner around speech-to-speech that bounds speculative VAD turns.

In speech-to-speech 1.0.0:
1. VADHandler._should_reopen_current_turn keeps concatenating _speculative_audio_prefix
   indefinitely while a turn is uncommitted, causing O(T^2) Parakeet memory growth
   during rapid table speech.
2. LMOutputProcessor.process never commits speculative_turns when an LLM response
   contains only tool calls (e.g. idle_do_nothing on IGNORE turns) and no TTSInput,
   leaving every IGNORE turn uncommitted forever.
3. Chat.has_pending_tool_calls refuses every later LLM request while any function_call
   lacks its output, with no timeout, so one robot gesture result the app never sends
   silences the familiar until reconnect.
4. One brain request could hold the LLM thread for about a minute: the SDK retried a
   timed-out request twice and nothing bounded a slow stream.
5. A wedged handler leaves its pool unit "stuck" forever while /v1/pool still answers 200,
   so every reconnect gets session_limit_reached. A watchdog thread exits with code 70
   so start_all.sh restarts s2s.

This wrapper applies surgical runtime guards before launching speech_to_speech.cli.main().
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

# Run as a script from the s2s venv, so the repo root is not on sys.path by default.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# pool_is_wedged is re-exported here; the wake daemon imports it from familiar_brain.
from familiar_brain.stack_health import DEFAULT_STUCK_EXIT_S, pool_is_wedged  # noqa: E402

logger = logging.getLogger("run_s2s")

DEFAULT_MAX_TURN_AUDIO_S = 30.0
DEFAULT_MAX_TURN_REVISIONS = 8
DEFAULT_LLM_DEADLINE_S = 20.0
KOKORO_VOICE_RE = re.compile(r"^[abefhijpz][fm]_[a-z][a-z0-9_]*$")
# Robot-side tools that only move Rocky (robot_profile/familiar/profile.md, app.py
# GESTURE_TOOLS). An unanswered call to one of these must not block later turns.
ROBOT_GESTURE_TOOLS = frozenset({"play_emotion", "stop_emotion", "head_tracking", "move_head", "idle_do_nothing"})


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def configure_cuda_env() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def configure_espeak_env() -> None:
    """Populate phonemizer/espeak paths from espeakng_loader if installed (no sudo needed)."""
    try:
        import espeakng_loader  # type: ignore[import-not-found]
    except ImportError:
        return
    try:
        lib_path = espeakng_loader.get_library_path()
        data_path = espeakng_loader.get_data_path()
    except Exception:
        return
    if lib_path:
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(lib_path))
    if data_path:
        os.environ.setdefault("ESPEAK_DATA_PATH", str(data_path))


def patch_vad_handler(
    vad_cls: type[Any],
    *,
    max_turn_audio_s: float | None = None,
    max_turn_revisions: int | None = None,
) -> bool:
    """Bound cumulative speculative audio and reopen count on VADHandler."""
    if getattr(vad_cls._should_reopen_current_turn, "_familiar_runaway_patched", False):
        return False

    orig_should_reopen = vad_cls._should_reopen_current_turn

    @functools.wraps(orig_should_reopen)
    def bounded_should_reopen(self: Any, audio_start_ms: int) -> bool:
        if not orig_should_reopen(self, audio_start_ms):
            return False

        audio_cap_s = (
            max_turn_audio_s
            if max_turn_audio_s is not None
            else float(
                getattr(self, "max_turn_audio_s", None)
                or _env_float("S2S_MAX_TURN_AUDIO_S", DEFAULT_MAX_TURN_AUDIO_S)
            )
        )
        rev_cap = (
            max_turn_revisions
            if max_turn_revisions is not None
            else int(
                getattr(self, "max_turn_revisions", None)
                or _env_int("S2S_MAX_TURN_REVISIONS", DEFAULT_MAX_TURN_REVISIONS)
            )
        )

        prefix = getattr(self, "_speculative_audio_prefix", None)
        sample_rate = getattr(self, "sample_rate", 16000) or 16000
        if prefix is not None and audio_cap_s > 0 and len(prefix) >= int(sample_rate * audio_cap_s):
            logger.info(
                "VAD: capping turn %s (prefix=%.2fs >= %.2fs); starting fresh turn",
                getattr(self, "_current_turn_id", None),
                len(prefix) / sample_rate,
                audio_cap_s,
            )
            return False

        rev = getattr(self, "_current_turn_revision", None)
        if rev is not None and rev_cap > 0 and int(rev) >= rev_cap:
            logger.info(
                "VAD: capping turn %s (revision=%s >= %s); starting fresh turn",
                getattr(self, "_current_turn_id", None),
                rev,
                rev_cap,
            )
            return False

        return True

    bounded_should_reopen._familiar_runaway_patched = True  # type: ignore[attr-defined]
    vad_cls._should_reopen_current_turn = bounded_should_reopen
    return True


def patch_lm_output_processor(lm_proc_cls: type[Any]) -> bool:
    """Commit speculative turns when an EndOfResponse completes with tool calls only."""
    if getattr(lm_proc_cls.process, "_familiar_commit_patched", False):
        return False

    orig_process = lm_proc_cls.process

    @functools.wraps(orig_process)
    def committing_process(self: Any, lm_output: Any) -> Iterator[Any]:
        is_end_of_response = (
            hasattr(lm_output, "error")
            and hasattr(lm_output, "turn_id")
            and hasattr(lm_output, "turn_revision")
            and type(lm_output).__name__ == "EndOfResponse"
        )
        had_tool_calls = bool(getattr(self, "_tool_call_ids", None)) if is_end_of_response else False
        turn_id = getattr(lm_output, "turn_id", None) if is_end_of_response else None
        turn_revision = getattr(lm_output, "turn_revision", None) if is_end_of_response else None

        saw_response_done = False
        for item in orig_process(self, lm_output):
            if is_end_of_response and type(item).__name__ == "AssistantResponseDoneEvent":
                saw_response_done = True
            yield item

        if is_end_of_response and saw_response_done and had_tool_calls:
            speculative_turns = getattr(self, "speculative_turns", None)
            if speculative_turns is not None:
                speculative_turns.commit(turn_id, turn_revision)

    committing_process._familiar_commit_patched = True  # type: ignore[attr-defined]
    lm_proc_cls.process = committing_process
    return True


def patch_chat_pending_tools(chat_cls: type[Any]) -> bool:
    """Let later turns generate while only robot gesture calls await their output.

    Chat.has_pending_tool_calls is the gate every LLM request and response.create checks.
    A gesture needs no reply and the app may never send its result, so it must not hold
    the conversation. Its late output is still accepted by append_tool_output.
    """
    if getattr(chat_cls.has_pending_tool_calls, "_familiar_pending_patched", False):
        return False

    orig_has_pending = chat_cls.has_pending_tool_calls

    @functools.wraps(orig_has_pending)
    def has_blocking_pending_tool_calls(self: Any) -> bool:
        with self._lock:
            return any(
                getattr(fc, "name", None) not in ROBOT_GESTURE_TOOLS
                for fc in self._pending_tool_calls.values()
            )

    has_blocking_pending_tool_calls._familiar_pending_patched = True  # type: ignore[attr-defined]
    chat_cls.has_pending_tool_calls = has_blocking_pending_tool_calls
    return True


def patch_responses_request(lm_cls: type[Any]) -> bool:
    """Bound one brain request: no SDK retries, and a total deadline on the stream.

    Each retry cost another httpx read timeout, and nothing could interrupt a slow stream,
    so one turn held the LLM thread for about a minute. At the deadline the stream is
    closed and httpx.ReadTimeout raised, which s2s's _generate already turns into its
    fallback reply and an EndOfResponse.
    """
    if getattr(lm_cls._request, "_familiar_deadline_patched", False):
        return False

    orig_request = lm_cls._request
    orig_iter = lm_cls._iter_stream_events

    @functools.wraps(orig_request)
    def no_retry_request(self: Any, api_input: Any, optional_kwargs: dict[str, Any]) -> Any:
        if self.client.max_retries != 0:
            self.client = self.client.with_options(max_retries=0)
        return orig_request(self, api_input, optional_kwargs)

    @functools.wraps(orig_iter)
    def deadline_iter(self: Any, api_response: Any) -> Iterator[Any]:
        import httpx

        deadline_s = _env_float("FAMILIAR_LLM_DEADLINE_S", DEFAULT_LLM_DEADLINE_S)
        expired = threading.Event()
        lock = threading.Lock()
        finished = False

        def expire() -> None:
            with lock:  # never close a stream this generator has already let go of
                if finished:
                    return
                expired.set()
            logger.warning("LLM stream passed its %.1fs deadline; closing it", deadline_s)
            api_response.close()

        timer = threading.Timer(deadline_s, expire)
        timer.daemon = True
        timer.start()
        try:
            for event in orig_iter(self, api_response):
                if expired.is_set():
                    break
                yield event
        except Exception as exc:
            if expired.is_set():
                raise httpx.ReadTimeout(f"LLM stream passed its {deadline_s:.1f}s deadline") from exc
            raise
        finally:
            with lock:
                finished = True
            timer.cancel()
        if expired.is_set():
            raise httpx.ReadTimeout(f"LLM stream passed its {deadline_s:.1f}s deadline")

    no_retry_request._familiar_deadline_patched = True  # type: ignore[attr-defined]
    lm_cls._request = no_retry_request
    lm_cls._iter_stream_events = deadline_iter
    return True


def is_valid_kokoro_voice(voice: str | None) -> bool:
    return bool(voice and KOKORO_VOICE_RE.match(voice.strip()))


def _read_override_voice() -> str | None:
    """Read live Kokoro voice override written by POST /api/voice (data/voice.json)."""
    voice_file = os.environ.get("FAMILIAR_VOICE_FILE", "data/voice.json")
    try:
        import json
        from pathlib import Path

        p = Path(voice_file)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        v = data.get("voice") if isinstance(data, dict) else None
        if is_valid_kokoro_voice(v):
            return str(v).strip()
    except Exception:
        return None
    return None


def patch_kokoro_voice_guard(kokoro_cls: type[Any]) -> bool:
    """Prevent non-Kokoro Realtime session voice overrides (e.g. 'Aiden') from crashing KPipeline,
    and honor live Kokoro voice changes written to data/voice.json without restarting s2s."""
    patched = False
    for attr in ("_process_kokoro", "_process_mlx"):
        orig_fn = getattr(kokoro_cls, attr, None)
        if orig_fn is None or getattr(orig_fn, "_familiar_voice_guard_patched", False):
            continue

        @functools.wraps(orig_fn)
        def guarded(self: Any, llm_sentence: str, language_code: str | None = None, _orig: Any = orig_fn) -> Iterator[Any]:
            override = _read_override_voice()
            if override:
                self.voice = override
                # Kokoro voice prefix ('a' American, 'b' British) selects language pipeline.
                if override[0] in ("a", "b") and language_code is None:
                    language_code = override[0]
            current_voice = getattr(self, "voice", None)
            if not is_valid_kokoro_voice(current_voice):
                fallback = getattr(self, "_initial_voice", None)
                if not is_valid_kokoro_voice(fallback):
                    fallback = "bm_fable"
                logger.warning(
                    "Kokoro: ignoring non-Kokoro session voice %r; keeping %r",
                    current_voice,
                    fallback,
                )
                self.voice = fallback
            yield from _orig(self, llm_sentence, language_code)

        guarded._familiar_voice_guard_patched = True  # type: ignore[attr-defined]
        setattr(kokoro_cls, attr, guarded)
        patched = True
    return patched


def apply_s2s_patches(
    *,
    max_turn_audio_s: float | None = None,
    max_turn_revisions: int | None = None,
) -> None:
    from speech_to_speech.LLM.chat import Chat  # type: ignore[import-not-found]
    from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor  # type: ignore[import-not-found]
    from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler  # type: ignore[import-not-found]
    from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler  # type: ignore[import-not-found]
    from speech_to_speech.VAD.vad_handler import VADHandler  # type: ignore[import-not-found]

    patch_vad_handler(
        VADHandler,
        max_turn_audio_s=max_turn_audio_s,
        max_turn_revisions=max_turn_revisions,
    )
    patch_lm_output_processor(LMOutputProcessor)
    patch_chat_pending_tools(Chat)
    patch_responses_request(ResponsesApiModelHandler)
    patch_kokoro_voice_guard(KokoroTTSHandler)


WEDGED_EXIT_CODE = 70
POOL_POLL_INTERVAL_S = 5.0


def pool_wedged_now(url: str, max_drain_s: float, timeout_s: float = POOL_POLL_INTERVAL_S) -> bool:
    """One /v1/pool poll. Unreachable (models still loading) counts as not wedged."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            pool = json.load(resp)
    except OSError:
        return False
    except ValueError as exc:
        logger.warning("pool watchdog: /v1/pool returned bad JSON: %s", exc)
        return False
    if pool_is_wedged(pool, max_drain_s):
        logger.warning("pool watchdog: s2s pool is wedged: %s", pool.get("units") if isinstance(pool, dict) else pool)
        return True
    return False


def watch_pool(
    url: str,
    *,
    max_drain_s: float,
    interval_s: float = POOL_POLL_INTERVAL_S,
    exit_fn: Callable[[int], Any] = os._exit,
) -> None:
    while True:
        time.sleep(interval_s)
        try:
            wedged = pool_wedged_now(url, max_drain_s, timeout_s=interval_s)
        except Exception:
            # e.g. http.client.IncompleteRead; the watchdog must outlive any one bad poll.
            logger.warning("pool watchdog: poll failed", exc_info=True)
            continue
        if wedged:
            logger.warning("pool watchdog: exiting with code %d so start_all.sh restarts s2s", WEDGED_EXIT_CODE)
            # os._exit skips logging's flush at exit, so flush the WARNING into s2s.log first.
            for handler in logging.getLogger().handlers:
                handler.flush()
            sys.stderr.flush()
            exit_fn(WEDGED_EXIT_CODE)


def _serve_port(args: list[str]) -> int:
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            return int(args[i + 1])
        if arg.startswith("--port="):
            return int(arg.split("=", 1)[1])
    return 8765  # speech_to_speech realtime server default


def start_pool_watchdog(args: list[str]) -> threading.Thread:
    url = f"http://127.0.0.1:{_serve_port(args)}/v1/pool"
    max_drain_s = _env_float("S2S_STUCK_EXIT_S", DEFAULT_STUCK_EXIT_S)
    thread = threading.Thread(
        target=watch_pool, args=(url,), kwargs={"max_drain_s": max_drain_s}, daemon=True, name="pool-watchdog"
    )
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    configure_cuda_env()
    configure_espeak_env()
    apply_s2s_patches()
    start_pool_watchdog(list(sys.argv[1:] if argv is None else argv))

    from speech_to_speech.cli import main as s2s_main  # type: ignore[import-not-found]

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    result = s2s_main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
