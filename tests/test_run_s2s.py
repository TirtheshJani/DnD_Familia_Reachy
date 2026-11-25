from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_s2s


class _FakeSpeculativeTurns:
    def __init__(self) -> None:
        self.commits: list[tuple[str | None, int | None]] = []

    def commit(self, turn_id: str | None, revision: int | None) -> None:
        self.commits.append((turn_id, revision))


def _make_vad_cls(orig_result: bool = True):
    class FakeVAD:
        sample_rate = 16000

        def __init__(self, prefix_samples: int = 0, revision: int = 0) -> None:
            self._current_turn_id = "turn_1"
            self._current_turn_revision = revision
            self._speculative_audio_prefix = (
                np.zeros(prefix_samples, dtype=np.float32) if prefix_samples > 0 else None
            )

        def _should_reopen_current_turn(self, audio_start_ms: int) -> bool:
            return orig_result

    return FakeVAD


def test_patch_vad_handler_caps_audio_and_revisions_idempotently(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S2S_MAX_TURN_AUDIO_S", "30")
    monkeypatch.setenv("S2S_MAX_TURN_REVISIONS", "8")
    FakeVAD = _make_vad_cls(orig_result=True)

    assert run_s2s.patch_vad_handler(FakeVAD) is True
    assert run_s2s.patch_vad_handler(FakeVAD) is False

    # Under both caps -> allowed
    ok = FakeVAD(prefix_samples=16000 * 10, revision=2)
    assert ok._should_reopen_current_turn(1000) is True

    # At audio cap (30s) -> rejected
    at_audio_cap = FakeVAD(prefix_samples=16000 * 30, revision=1)
    assert at_audio_cap._should_reopen_current_turn(1000) is False

    # At revision cap (8) -> rejected
    at_rev_cap = FakeVAD(prefix_samples=16000 * 5, revision=8)
    assert at_rev_cap._should_reopen_current_turn(1000) is False


def test_patch_vad_handler_respects_original_false() -> None:
    FakeVAD = _make_vad_cls(orig_result=False)
    assert run_s2s.patch_vad_handler(FakeVAD, max_turn_audio_s=30.0, max_turn_revisions=8) is True
    inst = FakeVAD(prefix_samples=16000, revision=0)
    assert inst._should_reopen_current_turn(500) is False


def test_patch_lm_output_processor_commits_tool_only_end_of_response() -> None:
    class EndOfResponse:
        def __init__(self, turn_id: str, turn_revision: int, error: str | None = None) -> None:
            self.turn_id = turn_id
            self.turn_revision = turn_revision
            self.error = error

    class AssistantResponseDoneEvent:
        pass

    class ResponseFailedEvent:
        pass

    class FakeLMOutputProcessor:
        def __init__(self, *, succeed: bool = True, stale: bool = False) -> None:
            self.speculative_turns = _FakeSpeculativeTurns()
            self._tool_call_ids: list[str] = []
            self._succeed = succeed
            self._stale = stale

        def process(self, lm_output):
            if self._stale:
                self._tool_call_ids = []
                yield EndOfResponse(lm_output.turn_id, lm_output.turn_revision)
                return
            if self._succeed:
                yield AssistantResponseDoneEvent()
            else:
                yield ResponseFailedEvent()
            yield lm_output
            self._tool_call_ids = []

    assert run_s2s.patch_lm_output_processor(FakeLMOutputProcessor) is True
    assert run_s2s.patch_lm_output_processor(FakeLMOutputProcessor) is False

    # Tool-only IGNORE response -> commits after AssistantResponseDoneEvent even though _reset cleared _tool_call_ids
    proc = FakeLMOutputProcessor(succeed=True)
    proc._tool_call_ids = ["call_idle_1"]
    items = list(proc.process(EndOfResponse("turn_9", 2)))
    assert len(items) == 2
    assert proc.speculative_turns.commits == [("turn_9", 2)]

    # Stale turn -> no commit
    stale_proc = FakeLMOutputProcessor(stale=True)
    stale_proc._tool_call_ids = ["call_idle_2"]
    list(stale_proc.process(EndOfResponse("turn_9", 1)))
    assert stale_proc.speculative_turns.commits == []

    # Errored turn -> no commit
    err_proc = FakeLMOutputProcessor(succeed=False)
    err_proc._tool_call_ids = ["call_idle_3"]
    list(err_proc.process(EndOfResponse("turn_9", 3, error="boom")))
    assert err_proc.speculative_turns.commits == []

    # Non-tool turn -> LMOutputProcessor patch does not double-commit
    plain_proc = FakeLMOutputProcessor(succeed=True)
    plain_proc._tool_call_ids = []
    list(plain_proc.process(EndOfResponse("turn_10", 0)))
    assert plain_proc.speculative_turns.commits == []


def test_configure_cuda_env_sets_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    run_s2s.configure_cuda_env()
    assert run_s2s.os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    run_s2s.configure_cuda_env()
    assert run_s2s.os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_patch_kokoro_voice_guard_blocks_invalid_override(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAMILIAR_VOICE_FILE", str(tmp_path / "voice.json"))
    class FakeKokoro:
        def __init__(self) -> None:
            self._initial_voice = "bm_fable"
            self.voice = "bm_fable"
            self.seen_voices: list[str] = []

        def _process_kokoro(self, llm_sentence: str, language_code=None):
            self.seen_voices.append(self.voice)
            yield llm_sentence

    assert run_s2s.patch_kokoro_voice_guard(FakeKokoro) is True
    assert run_s2s.patch_kokoro_voice_guard(FakeKokoro) is False

    k = FakeKokoro()
    k.voice = "Aiden"  # invalid Kokoro voice override from Realtime session
    list(k._process_kokoro("Hello"))
    assert k.voice == "bm_fable"
    assert k.seen_voices == ["bm_fable"]

    k.voice = "bm_george"  # valid Kokoro voice override allowed
    list(k._process_kokoro("Hello again"))
    assert k.seen_voices[-1] == "bm_george"


@pytest.mark.skipif(
    importlib.util.find_spec("speech_to_speech") is None,
    reason="speech_to_speech package only installed in WSL2 s2s venv",
)
def test_real_s2s_vad_and_lm_processor_bound_continuous_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    from queue import Queue
    from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
    from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
    from speech_to_speech.pipeline.messages import AssistantToolCallPart, EndOfResponse, LLMResponseChunk
    from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
    from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
    from speech_to_speech.VAD.vad_handler import VADHandler

    monkeypatch.setenv("S2S_MAX_TURN_AUDIO_S", "30")
    monkeypatch.setenv("S2S_MAX_TURN_REVISIONS", "8")
    run_s2s.apply_s2s_patches(max_turn_audio_s=30.0, max_turn_revisions=8)

    tracker = SpeculativeTurnTracker()
    vad = VADHandler.__new__(VADHandler)
    vad.sample_rate = 16000
    vad.speculative_turns = tracker
    vad.unanswered_reopen_ms = 1500
    vad.speculative_reopen_ms = 1200
    vad.streaming_stt_sink = None
    vad._pending_reopen_candidate = None
    vad._speech_started_emitted = False
    vad._current_turn_id = None
    vad._current_turn_revision = None
    vad._last_final_audio_ms = None
    vad._speculative_audio_prefix = None
    vad._speculative_raw_audio_prefix = None
    vad._turn_counter = 0

    # Simulate 24 consecutive 5s segments separated by 300ms pauses (120s total speech)
    # without any assistant commit: turns MUST split so prefix never exceeds 30s!
    turn_ids: list[str] = []
    max_prefix_s = 0.0
    t_ms = 0
    for _ in range(24):
        seg = np.zeros(16000 * 5, dtype=np.float32)
        turn_id, rev, _reopened = vad._ensure_turn_for_speech_start(t_ms)
        combined = vad._combined_turn_audio(seg)
        max_prefix_s = max(max_prefix_s, len(combined) / 16000.0)
        vad._speculative_audio_prefix = combined
        vad._last_final_audio_ms = t_ms + 5000
        vad._speech_started_emitted = False
        turn_ids.append(turn_id)
        t_ms += 5300  # 300ms pause < unanswered_reopen_ms

    assert max_prefix_s <= 35.0
    assert len(set(turn_ids)) >= 4

    # Verify LMOutputProcessor commits tool-only IGNORE turns cleanly
    proc = LMOutputProcessor(stop_event=SimpleNamespace(is_set=lambda: False), queue_in=Queue(), queue_out=Queue())
    proc.setup(speculative_turns=tracker, text_output_queue=Queue())
    tracker.observe("turn_ignore", 0)
    chunk = LLMResponseChunk(
        parts=[
            AssistantToolCallPart(
                tool=ResponseFunctionToolCall(
                    call_id="call_1",
                    name="idle_do_nothing",
                    arguments="{}",
                    type="function_call",
                )
            )
        ],
        turn_id="turn_ignore",
        turn_revision=0,
    )
    list(proc.process(chunk))
    list(proc.process(EndOfResponse(turn_id="turn_ignore", turn_revision=0)))
    assert tracker.is_committed("turn_ignore", 0) is True
    assert getattr(KokoroTTSHandler._process_kokoro, "_familiar_voice_guard_patched", False) is True
