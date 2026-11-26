from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from familiar_brain import indexing, learning, modes, tools
from familiar_brain.speaker_id import (
    InsufficientSpeechError,
    SpeakerRegistry,
    make_speaker_embedder,
)
from familiar_brain.attention import Attention
from familiar_brain.behavior import BehaviorSettings
from familiar_brain.companion_api import (
    EventBus,
    VoiceStore,
    initiative_payload,
    register_companion_routes,
)
from familiar_brain.campaign import CAMPAIGN_FILE, read_body, refresh_campaign, save_manual_campaign
from familiar_brain.config import load_config
from familiar_brain.gate import Decision, Gate
from familiar_brain.knowledge import build_context
from familiar_brain.lexicon import Lexicon
from familiar_brain.live_index import LiveIndexer
from familiar_brain.llm import REACTION_INSTRUCTION, LLMClient
from familiar_brain.motion import (
    REACTION_MOODS,
    Choreographer,
    addressed_mood,
    sleep_transition,
    table_mood,
    tracking_item,
)
from familiar_brain.persona import load_persona, render_system_prompt
from familiar_brain.profiles import CampaignManager
from familiar_brain.responses_api import (
    IDLE_TOOL,
    continuation_kind,
    empty_response_events,
    idle_items,
    idle_response_events,
    idle_response_json,
    item_events,
    latest_user_text,
    response_json,
    sse,
    tool_call_events,
    wants_stream,
    with_instructions_and_tools,
)
from familiar_brain.session import SessionLog
from familiar_brain.summariser import (
    Paused,
    Purged,
    apply_retention,
    checkpoint_session,
    load_players,
    purge_session,
    validate_session_id,
)
from familiar_brain.state import BrainState
from familiar_brain.timing import TurnTimer
from familiar_brain.tools import (
    behavior_setting as behavior_tool,
    initiative as initiative_tool,
    lookup_rule as lookup_tool,
    note as note_tool,
    recall as recall_tool,
    roll as roll_tool,
)
from familiar_brain.upstream import Upstream

# The persona moods played as cues when the brain is muted and unmuted.
ASLEEP, AWAKE = "asleep", "awake"

# Robot-side tools that only move Rocky: the ones the brain's preface sends (motion.py,
# attention.py), plus stop_emotion, the other gesture tool the app offers the model.
GESTURE_TOOLS = frozenset({"play_emotion", "head_tracking", "move_head", "stop_emotion", IDLE_TOOL})

# How soon after an utterance a request with the same history can still be a revision of
# it. speech-to-speech reopens a turn after a pause of up to 7 s (unanswered_reopen_ms),
# and the player may talk on for a while before the revision is sent. Past this, the
# same history is more likely a new connection starting from the same system message.
REVISION_WINDOW_S = 30.0


@dataclass(frozen=True)
class _PrefaceStart:
    """The one-shot state a preface spends, as it stood before the preface ran."""

    tracking: bool
    gesture_at: float
    reaction_at: float
    cue: str | None


def _without_gestures(req: dict, *, hide_camera: bool = False) -> dict:
    """The request with every gesture and idle call, and its result, out of the history,
    and the gesture tools out of the tools it offers.

    speech-to-speech records each call the brain sends ahead of the reply text, so from the
    second turn on the history shows a model that answers with play_emotion before it
    speaks. Gemma copies that. Against llama-server on 2026-09-16, with three earlier
    addressed turns in the history, 5 of 6 replies to an addressed line were a bare
    play_emotion call with no text, against 0 of 6 with the gesture pairs left out.
    play_emotion needs no response, so Rocky gestured and said nothing. idle_do_nothing
    goes too: it is the same brain-made call with no speech after it. Camera calls, their
    results and the frame stay, or sight loses its image. Only the copy sent upstream
    changes: continuation_kind reads the request as it came, so an echo is still an echo.

    The tools go for the same reason. With the history already clean, 16 addressed lines
    on 2026-09-16 with the robot's tools offered got 2 bare move_head calls and 4 replies
    that spoke the call as text ("play_emotion(thinking)", "I_head(left)"). Hidden, none
    of either: 12 spoke and 4 called the camera for "look at me". The brain sends every
    gesture itself (motion.py, attention.py). Only this copy changes, so speech-to-speech
    and the app still hold the full tool list the brain's own gesture calls rely on.
    """
    hidden_tools = GESTURE_TOOLS | {"camera"} if hide_camera else GESTURE_TOOLS
    offered = req.get("tools")
    if isinstance(offered, list):
        req = {**req, "tools": [t for t in offered
                                if not (isinstance(t, dict) and t.get("name") in hidden_tools)]}
    items = req.get("input")
    if not isinstance(items, list):
        return req

    def is_gesture_call(item) -> bool:
        return (isinstance(item, dict) and item.get("type") == "function_call"
                and item.get("name") in GESTURE_TOOLS)

    call_ids = {i.get("call_id") for i in items if is_gesture_call(i)} - {None}
    kept = [i for i in items if not is_gesture_call(i)
            and not (isinstance(i, dict) and i.get("type") == "function_call_output"
                     and i.get("call_id") in call_ids)]
    return {**req, "input": kept}


def _history_key(req: dict) -> str | None:
    """A digest of everything before the request's closing user message, used to recognise
    a revision of the previous utterance.

    None when the request does not close with a user message, or has nothing before it.
    speech-to-speech always opens with the session's system message, so a bare message is
    a client that sends no history (a probe, a test), where every line would otherwise
    look like a revision of the one before.
    """
    items = req.get("input")
    if not (isinstance(items, list) and len(items) > 1 and isinstance(items[-1], dict)
            and items[-1].get("role") == "user"):
        return None
    return hashlib.sha256(json.dumps(items[:-1], sort_keys=True, default=str).encode()).hexdigest()


def _correct_user_text(req: dict, lexicon: Lexicon) -> str:
    """Rewrite the last user item with canonical setting spellings, and return it.

    Correcting the request itself rather than only the extracted string means the gate,
    the session log, retrieval and the model all see the same words.
    Thraka and Nephria come off the wire as whatever the STT heard, "Thraca" and
    "Nefrea"; a name that reaches retrieval misspelt finds nothing, and one that reaches
    the model misspelt gets spoken back wrong.
    """
    text = lexicon.correct(latest_user_text(req))
    for item in reversed(req.get("input") or []):
        if not (isinstance(item, dict) and item.get("role") == "user"):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            item["content"] = text
            return text
        # latest_user_text joins every text part, so the corrected string stands for all
        # of them: fold it into the first and drop the rest, keeping any non-text parts.
        out, folded = [], False
        for part in content:
            is_text = isinstance(part, dict) and part.get("type") in ("input_text", "text")
            if is_text and not folded:
                out.append({**part, "text": text})
                folded = True
            elif not is_text:
                out.append(part)
        item["content"] = out if folded else content
        return text
    return text


STATIC_DIR = Path(__file__).resolve().parent / "static"


def _no_robot_env() -> bool:
    return os.environ.get("FAMILIAR_NO_ROBOT", "").strip().lower() in ("1", "true", "yes", "on")


def _native_robot_env() -> bool:
    return os.environ.get("FAMILIAR_ROBOT_MODE", "").strip().lower() == "native"


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    return host.strip().lower() in ("127.0.0.1", "localhost", "::1")


def _load_robot_connect() -> Any:
    try:
        from scripts import robot_connect  # type: ignore[import-not-found]
        return robot_connect
    except ImportError:
        rc_path = Path(__file__).resolve().parent.parent / "scripts" / "robot_connect.py"
        if not rc_path.exists():
            return None
        spec = importlib.util.spec_from_file_location("scripts.robot_connect", rc_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def create_app(config_path: Path | str = "config/brain.toml") -> Flask:
    cfg = load_config(config_path)
    persona = load_persona(cfg.paths.persona)
    lexicon = Lexicon.load(cfg.paths.lexicon)
    state = BrainState()
    data_dir = Path(cfg.knowledge.db_path).parent
    campaigns = CampaignManager(
        data_dir=data_dir,
        default_notes_dir=Path(cfg.knowledge.notes_dir),
        default_sessions_dir=Path(cfg.paths.sessions_dir),
        default_speakers_path=Path(cfg.speakers.store_path),
        default_players_path=Path(cfg.paths.players),
    )
    active_prof = campaigns.active()
    log = SessionLog(active_prof.sessions_dir)
    llm = LLMClient(base_url=cfg.upstream.base_url, classify_slot=cfg.upstream.classify_slot,
                    classify_timeout_s=cfg.upstream.classify_timeout_s,
                    generate_slot=cfg.upstream.generate_slot,
                    classify_cooldown_s=cfg.upstream.classify_cooldown_s)
    gate = Gate(state=state, name=persona.name, aliases=persona.aliases,
                follow_up_window_s=cfg.gate.follow_up_window_s,
                max_follow_ups=cfg.gate.max_follow_ups,
                classifier=(lambda segs: llm.classify(segs, persona_name=persona.name))
                if cfg.gate.llm_classifier else None)
    store = indexing.open_store(cfg)
    if active_prof.notes_dir.exists():
        indexing.index_notes(store, active_prof.notes_dir)
    behavior = BehaviorSettings(data_dir / "behavior.json", defaults=asdict(cfg.behavior))
    voice_store = VoiceStore(data_dir / "voice.json", default_voice=persona.tts_speaker or "bm_fable")
    event_bus = EventBus()
    for mod in (roll_tool, initiative_tool, note_tool, lookup_tool, recall_tool, behavior_tool):
        tools.register(mod)
    note_tool.bind(log)
    lookup_tool.bind(store)
    recall_tool.bind(store)
    learning.warn_if_no_players(active_prof.players_path)
    learner = learning.Learner(active_prof.sessions_dir, store=store, llm=llm, players_path=active_prof.players_path,
                               active=lambda: log.active_id)
    live = LiveIndexer(store)
    # Catch-up is queued before retention runs, and retention leaves whatever the learner
    # holds, so a session that was never ended cannot lose its raw log unlearnt.
    learner.catch_up()
    apply_retention(active_prof.sessions_dir, store=store, retention_days=cfg.knowledge.retention_days,
                    keep=learner.busy())
    upstream = Upstream(base_url=cfg.upstream.base_url, generate_slot=cfg.upstream.generate_slot,
                        brain_tool_names={s["name"] for s in tools.specs()})
    # No mood call on addressed turns: it ran ahead of the first word, and live on
    # 2026-09-16 it timed out on 11 of 11 addressed turns under GPU contention, always
    # falling back to attentive, the emotion that played over Rocky's speech. The keyword
    # rules alone choose a reply's gesture (motion.for_addressed).
    choreo = Choreographer(
        persona, min_gap_s=cfg.motion.min_gesture_gap_s, reaction_gap_s=cfg.motion.reaction_gap_s,
        clock=time.time,
        mood_classifier=None,
        reaction_classifier=(lambda t: llm.mood(t, moods=REACTION_MOODS + ("none",), prompt=REACTION_INSTRUCTION))
        if cfg.motion.llm_reaction else None,
        behavior_provider=behavior.snapshot,
    )
    attention = Attention(state=state)
    checkpoint_lock = threading.Lock()
    speaker_embedder = make_speaker_embedder(
        cfg.speakers.backend, onnx_model_path=cfg.speakers.onnx_model_path
    )
    speaker_registry = SpeakerRegistry(active_prof.speakers_path, embedder=speaker_embedder)
    speaker_lock = threading.Lock()
    speaker_state = {"capture_count": 0}

    def _is_capturing() -> bool:
        with speaker_lock:
            return speaker_state["capture_count"] > 0

    tap_http = requests.Session()
    robot_http = requests.Session()
    robot_lock = threading.Lock()
    is_native_boot = _native_robot_env()
    robot_state: dict[str, Any] = {
        "backend_connected": False if is_native_boot else None,
        "host": "reachy-mini.local" if is_native_boot else None,
        "last_checked_at": None,
        "detail": "waiting for reachy_familiar audio link..." if is_native_boot else "not checked",
        "disabled": False,
        "mode": "native" if is_native_boot else "legacy",
        "mic_level": 0.0,
    }

    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.extensions["brain_state"] = state  # exposed for tests and the companion UI
    app.extensions["live_index"] = live
    app.extensions["learner"] = learner
    app.extensions["store"] = store
    app.extensions["session_log"] = log
    app.extensions["speaker_registry"] = speaker_registry
    app.extensions["upstream"] = upstream
    app.extensions["persona"] = persona
    app.extensions["behavior"] = behavior
    app.extensions["voice_store"] = voice_store
    app.extensions["event_bus"] = event_bus
    app.extensions["campaign_manager"] = campaigns

    def _query_tap_current() -> tuple[str | None, float | None]:
        if not cfg.speakers.enabled or not speaker_registry.profiles:
            return None, None
        try:
            r = tap_http.get(
                f"{cfg.speakers.tap_url.rstrip('/')}/_tap/current",
                headers={"Connection": "close"},
                timeout=0.05,
            )
            if r.ok:
                data = r.json()
                role = data.get("role")
                updated_at = data.get("updated_at")
                if isinstance(role, str) and role.strip() and isinstance(updated_at, (int, float)):
                    if (time.time() - float(updated_at)) <= cfg.speakers.max_age_s:
                        raw_c = data.get("confidence")
                        conf = float(raw_c) if isinstance(raw_c, (int, float)) and math.isfinite(float(raw_c)) else None
                        return role.strip(), conf
        except Exception:
            pass
        return None, None

    def _notify_tap_reload(*, store_path: Path | None = None) -> None:
        if not cfg.speakers.enabled:
            return
        try:
            payload = {"store_path": str(store_path)} if store_path is not None else {}
            tap_http.post(
                f"{cfg.speakers.tap_url.rstrip('/')}/_tap/reload",
                json=payload,
                headers={"Connection": "close"},
                timeout=0.3,
            )
        except Exception:
            pass

    def _speaker_status() -> dict:
        return {
            "enabled": bool(cfg.speakers.enabled),
            "enrolled": speaker_registry.enrolled_roles(),
            "last_speaker": state.last_speaker,
            "last_confidence": state.last_speaker_confidence,
            "capturing": _is_capturing(),
        }

    app.extensions["speaker_status"] = _speaker_status

    def _query_tap_ambient() -> dict[str, Any]:
        default_amb = {
            "mood": "quiet",
            "energy": 0.0,
            "brightness": 0.0,
            "pulse": 0.0,
            "tempo_bpm": 0.0,
            "updated_at": None,
        }
        if not cfg.speakers.enabled:
            return default_amb
        try:
            r = tap_http.get(
                f"{cfg.speakers.tap_url.rstrip('/')}/_tap/ambient",
                headers={"Connection": "close"},
                timeout=0.05,
            )
            if r.ok:
                data = r.json()
                if isinstance(data, dict) and "mood" in data:
                    return data
        except Exception:
            pass
        return default_amb

    def _robot_status() -> dict[str, Any]:
        with robot_lock:
            return dict(robot_state)

    app.extensions["robot_status"] = _robot_status

    def close() -> None:
        tap_http.close()
        robot_http.close()
        upstream.close()
        live.close()
        learner.close()
        store.close()

    app.extensions["close"] = close

    def warmup(*, warm_llm: bool = True) -> dict[str, Any]:
        """Pre-warm embedder, retrieval, regex/SSE helpers, and optionally upstream streaming
        without mutating BrainState, Choreographer, Attention, or SessionLog."""
        timer = TurnTimer()
        warmup_text = "Rocky, roll a d20 for me"
        warmup_req = {
            "model": "familiar",
            "stream": True,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": warmup_text}],
                }
            ],
            "tools": [],
        }
        llm_warmed = False
        frames = 0
        with app.test_request_context("/v1/responses", method="POST", json=warmup_req):
            req_copy = copy.deepcopy(warmup_req)
            corrected = _correct_user_text(req_copy, lexicon)
            gate._named(corrected)
            modes.ENTER.search(corrected)
            modes.EXIT.search(corrected)
            addressed_mood(corrected)
            table_mood("the door creaks")
            timer.mark("gate")
            ctx = ""
            try:
                ctx = build_context(
                    store,
                    corrected,
                    chunks_per_corpus=cfg.knowledge.chunks_per_corpus,
                    max_chars=cfg.knowledge.max_context_chars,
                )
            except Exception as exc:
                app.logger.warning("embedder pre-warm skipped: %s", exc)
            timer.mark("retrieval")
            instructions = render_system_prompt(persona, ctx, mode="familiar")
            outgoing = with_instructions_and_tools(
                _without_gestures(req_copy),
                instructions=instructions,
                extra_tools=tools.specs(),
            )
            preface_frames = [sse(ev) for ev in item_events([tracking_item(True)])]
            frames += len(preface_frames)
            timer.mark("preface")
            if warm_llm:
                try:
                    for _frame in stream_with_context(upstream.stream(outgoing, must_speak=True)):
                        frames += 1
                    llm_warmed = True
                except Exception as exc:
                    app.logger.warning("upstream LLM warmup skipped: %s", exc)
            timer.mark("generation")
        rec = timer.finish(decision="WARMUP", spoke=False)
        return {
            "llm": llm_warmed,
            "frames": frames,
            "total_ms": rec["total_ms"],
        }

    app.extensions["warmup"] = warmup

    def _resolve_speaker() -> tuple[str | None, float | None]:
        hdr_role = (request.headers.get("X-Familiar-Speaker-Role") or "").strip()
        if hdr_role:
            conf: float | None = None
            raw_conf = request.headers.get("X-Familiar-Speaker-Confidence")
            if raw_conf is not None:
                try:
                    val = float(raw_conf)
                    if math.isfinite(val):
                        conf = val
                except (TypeError, ValueError):
                    conf = None
            return hdr_role, conf
        return _query_tap_current()

    def _log(
        text: str,
        decision: Decision,
        spoke: bool,
        *,
        speaker: str | None = None,
        speaker_confidence: float | None = None,
    ) -> None:
        if not state.logging_enabled:
            return
        state.record_transcript(text=text, speaker=speaker, decision=decision.value, spoke=spoke, ts=time.time())
        if log.active_id is not None:
            log.append(
                text=text,
                decision=decision.value,
                spoke=spoke,
                speaker=speaker,
                speaker_confidence=speaker_confidence,
            )
        try:
            event_bus.publish("state", _state_payload())
        except Exception:
            pass

    def _instructions(text: str, timer: TurnTimer, *, speaker_role: str | None = None) -> str:
        """Retrieve for this utterance and prepend the result to the system prompt.

        Only ever called on an ADDRESSED turn, so an ignored line costs no retrieval. A
        continuation passes no text, so it costs none either. The length block follows
        the reply mode (modes.py).
        """
        context = ""
        if text.strip():
            try:
                context = build_context(store, text, chunks_per_corpus=cfg.knowledge.chunks_per_corpus,
                                        max_chars=cfg.knowledge.max_context_chars)
            except Exception as e:  # a broken store must not cost the table a reply
                app.logger.warning("retrieval failed: %s", e)
        timer.mark("retrieval")
        snap = behavior.snapshot()
        return render_system_prompt(
            persona,
            context,
            mode=state.mode,
            speaker_role=speaker_role,
            behavior=snap,
            camera_enabled=snap.camera_enabled,
            campaign_profile=campaigns.active(),
        )

    def _preface(text: str, decision: Decision, now: float) -> list[dict]:
        """Robot-side tool calls that go out ahead of (or instead of) speech this turn.

        Order: attention first (a tracking toggle), then a pending cue (mute, unmute),
        then the mood of the reply on an addressed turn or a reaction to the table on any
        other; an AMBIENT line always gets one (motion.AMBIENT_FALLBACK_MOOD). Every turn
        kind runs this, so the first line of the night switches tracking on whatever the
        gate made of it. The tracking toggle leads so that on the turn
        after a mute the head stops following faces before the sleep gesture plays, and
        after an unmute follows them again before the wake gesture. A muted brain sends at
        most the tracking-off toggle and its cue: no reaction. A continuation passes no
        text, so it carries no mood or reaction.
        """
        items: list[dict] = []
        try:
            items += attention.items()
            cue = state.consume_cue()
            if cue:
                items += choreo.cue(cue, now)
            if state.muted or state.asleep or not text:
                return items
            if decision is Decision.ADDRESSED:
                items += choreo.for_addressed(text, now)
            else:
                items += choreo.for_table(text, now, ambient=decision is Decision.AMBIENT)
        except Exception as e:  # a gesture is never worth the reply it would cost
            app.logger.error("preface failed after %d item(s): %s", len(items), e)
        return items

    # The last utterance's history key, when it came, and the state its preface started
    # from. Only the streamed utterance path reads or writes it.
    last_utterance: dict = {"key": None, "at": float("-inf"), "start": None}

    def _preface_start() -> _PrefaceStart:
        cue = state.consume_cue()  # BrainState has no peek; put it straight back
        if cue:
            state.set_cue(cue)
        return _PrefaceStart(tracking=state.tracking, gesture_at=choreo.last_at,
                             reaction_at=choreo.last_reaction_at, cue=cue)

    def _rewind_if_revision(key: str | None, now: float) -> None:
        """Put back what a superseded version of this utterance spent, before its revision's
        preface runs.

        With smart_turn on, speech-to-speech asks for a reply at the player's first pause.
        If the player carries on inside its reopen window, it replaces that user message
        with a transcript of the longer audio, asks again, and drops everything the first
        response sent, gestures included. A preface spends state only once: a tracking
        toggle goes out only on a change, a cue only once, and a gesture starts its gap.
        Without this the revision the robot does get carries none of them, so Rocky never
        tracks at all, or keeps following faces after a mute.

        A revision has the same history before its last user message, because the message
        is replaced in place and user messages carry no id on the wire. The line after any
        other utterance has that utterance's own user message in its history, so it never
        matches. A cue set since (a mute or unmute) wins over the one being put back.
        """
        previous = last_utterance["start"]
        if (key is not None and key == last_utterance["key"] and previous is not None
                and now - last_utterance["at"] < REVISION_WINDOW_S):
            app.logger.info("revision of the previous utterance: preface state put back")
            state.tracking = previous.tracking
            choreo.last_at = previous.gesture_at
            choreo.last_reaction_at = previous.reaction_at
            if previous.cue:
                state.set_cue(state.consume_cue() or previous.cue)
        last_utterance.update(key=key, at=now, start=_preface_start())

    def _without_streaming(
        req: dict,
        text: str,
        decision: Decision,
        timer: TurnTimer,
        model: str,
        kind: str | None,
        *,
        speaker_role: str | None = None,
        speaker_conf: float | None = None,
    ):
        """The client calls /v1/responses without `stream` for its warmup probe and for
        history compaction. Both expect a JSON Response body, not an event stream.

        No preface on this path: a JSON body is not relayed to the robot as it arrives.
        A continuation (`kind` not None) is answered but never logged, see responses().
        """
        label = decision.value if kind is None else kind.upper()
        if decision is not Decision.ADDRESSED:
            if kind is None:
                _log(text, decision, spoke=False, speaker=speaker_role, speaker_confidence=speaker_conf)
            rec = timer.finish(decision=label, spoke=False)
            if kind is None:
                state.last_turn_ms = rec["total_ms"]
            return jsonify(idle_response_json(model=model))
        outgoing = with_instructions_and_tools(
            _without_gestures(req, hide_camera=not behavior.snapshot().camera_enabled),
            instructions=_instructions(text, timer, speaker_role=speaker_role),
            extra_tools=tools.specs(),
        )
        try:
            body, spoke = upstream.complete(outgoing), True
        except Exception as e:  # upstream down: degrade to silence
            app.logger.error("upstream failed (non-streaming): %s", e)
            body, spoke = idle_response_json(model=model), False
        # Deliberately no mark_spoke here. This path serves the client's warmup probe
        # and history compaction, neither of which is speech at the table, and marking
        # it would open the follow-up window before anyone has said anything.
        if kind is None:
            _log(text, decision, spoke=spoke, speaker=speaker_role, speaker_confidence=speaker_conf)
        timer.mark("generation")
        rec = timer.finish(decision=label, spoke=spoke)
        if kind is None:
            state.last_turn_ms = rec["total_ms"]
        return jsonify(body)

    @app.post("/v1/responses")
    def responses():
        req = request.get_json(force=True) or {}
        timer = TurnTimer()
        model = str(req.get("model") or "familiar")
        now = time.time()
        kind = continuation_kind(req)
        if kind == "compaction":
            # Not the table talking: passed through as it came, with no persona, brain tools,
            # gate, state or log. Gated, a transcript that named the familiar opened the
            # follow-up window; in persona, the summary came back as prose, not JSON.
            timer.finish(decision="COMPACTION", spoke=False)
            try:
                if wants_stream(req):
                    return Response(stream_with_context(upstream.stream(req)), content_type="text/event-stream")
                return jsonify(upstream.complete(req))
            except Exception as e:  # speech-to-speech keeps its history unchanged on failure
                app.logger.error("compaction passthrough failed: %s", e)
                return jsonify(response_json([], model=model))
        if kind == "echo":
            # The client handing back a robot-side tool's result. Nothing was said, so
            # nothing is gated, logged or spoken. The empty reply carries no tool call,
            # so the client has nothing to run and no reason to ask again.
            timer.finish(decision="ECHO", spoke=False)
            if not wants_stream(req):
                return jsonify(response_json([], model=model))
            body = "".join(sse(e) for e in empty_response_events())
            return Response(body, content_type="text/event-stream")
        speaker_role: str | None = None
        speaker_conf: float | None = None
        if kind == "sight":
            # The frame the robot just took. Not an utterance: no lexicon, no log, no
            # follow-up accounting, and the frame is never written anywhere (D5). Muted,
            # Deep Night, or Eyes Closed consent toggle blocks sight.
            text = ""
            decision = (
                Decision.IGNORE
                if (state.muted or state.asleep or not behavior.snapshot().camera_enabled)
                else Decision.ADDRESSED
            )
            timer.mark("gate")
        elif kind == "greeting":
            # The app's session-open prompt. Answer it, but do not open the follow-up
            # window: nobody at the table has said anything yet (compare deviation E7).
            text = ""
            decision = Decision.IGNORE if (state.muted or state.asleep) else Decision.ADDRESSED
            # It also opens a new app session, whose MovementManager starts with tracking
            # off. Toggles go out only on a change, so without this a brain still holding
            # tracking on from the last session never sends it again (deviation P16).
            state.tracking = False
            timer.mark("gate")
        else:
            speaker_role, speaker_conf = _resolve_speaker()
            if speaker_role is not None:
                state.last_speaker = speaker_role
                state.last_speaker_confidence = speaker_conf
            text = _correct_user_text(req, lexicon)
            can_voice_sleep = (
                speaker_role.strip().lower() == "dm"
                if speaker_role is not None
                else not speaker_registry.profiles
            )
            trans = sleep_transition(text) if can_voice_sleep else None
            if trans is True and not state.asleep:
                state.asleep = True
                state.end_conversation()
                if not state.muted:
                    state.set_cue(ASLEEP)
            elif trans is False and state.asleep:
                state.asleep = False
                state.disarm_speak_now()
                if not state.muted:
                    state.set_cue(AWAKE)
            state.remember(text)
            decision = gate.decide(state.recent_segments, speaker_role=speaker_role)
            timer.mark("gate")
            modes.update_mode(state, text, addressed=decision is Decision.ADDRESSED,
                              now=now, timeout_s=cfg.motion.mode_timeout_s)
            if decision is Decision.ADDRESSED:
                voice_cmd = behavior_tool.parse_voice_setting_command(text)
                if voice_cmd is not None:
                    k_dial, v_dial = voice_cmd
                    behavior.update(**{k_dial: v_dial})
        # SIGHT and GREETING in the timing line, so gate-report's latency figures, which
        # read ADDRESSED records only, never include a camera round trip or the greeting.
        label = decision.value if kind is None else kind.upper()
        if not wants_stream(req):
            return _without_streaming(
                req, text, decision, timer, model, kind,
                speaker_role=speaker_role, speaker_conf=speaker_conf,
            )
        if kind == "greeting" and decision is Decision.ADDRESSED:
            # Set here, where this turn's preface spends it. A JSON greeting has no
            # preface, so a cue set for it would play on whatever streamed turn came next.
            state.set_cue(AWAKE)
        if kind is None:
            _rewind_if_revision(_history_key(req), now)
        preface = _preface(text, decision, now)
        # How long the mood or reaction call took, which D2 and D9 say to watch live.
        timer.mark("preface")
        if decision is not Decision.ADDRESSED:
            if kind is None:
                _log(text, decision, spoke=False, speaker=speaker_role, speaker_confidence=speaker_conf)
            rec = timer.finish(decision=label, spoke=False)
            if kind is None:
                state.last_turn_ms = rec["total_ms"]
            body = "".join(sse(e) for e in tool_call_events(preface or idle_items()))
            return Response(body, content_type="text/event-stream")
        outgoing = with_instructions_and_tools(
            _without_gestures(req, hide_camera=not behavior.snapshot().camera_enabled),
            instructions=_instructions(text, timer, speaker_role=speaker_role),
            extra_tools=tools.specs(),
        )

        def gen():
            spoke = False
            # An SSE comment, ignored by the client's parser, so the response headers go
            # out now. speech-to-speech waits on them, and llama sends nothing before its
            # first token.
            yield ": ok\n\n"
            try:
                # The preface goes out before the first upstream frame, so no gesture
                # waits on generation. speech-to-speech turns each output_item.done that
                # carries a function_call into a ToolCall for the app
                # (docs/protocol-notes.md 2).
                for ev in item_events(preface):
                    yield sse(ev)
                for frame in upstream.stream(outgoing, must_speak=True):
                    if "response.output_text.delta" in frame:
                        spoke = True
                    yield frame
            except Exception as e:  # upstream down: degrade to silence
                app.logger.error("upstream failed: %s", e)
                for ev in idle_response_events():
                    yield sse(ev)
            finally:
                if kind is None:
                    if spoke:
                        state.mark_spoke()
                    _log(text, decision, spoke=spoke, speaker=speaker_role, speaker_confidence=speaker_conf)
                timer.mark("generation")
                rec = timer.finish(decision=label, spoke=spoke)
                if kind is None:
                    state.last_turn_ms = rec["total_ms"]

        return Response(stream_with_context(gen()), content_type="text/event-stream")

    def _state_payload() -> dict[str, Any]:
        modes.update_mode(state, "", addressed=False, now=time.time(),
                          timeout_s=cfg.motion.mode_timeout_s)
        speaker_fn = app.extensions.get("speaker_status")
        robot_fn = app.extensions.get("robot_status")
        return {
            "muted": state.muted,
            "asleep": state.asleep,
            "mode": state.mode,
            "tracking": state.tracking,
            "logging": state.logging_enabled,
            "session": log.active_id,
            "session_started_at": state.session_started_at,
            "last_spoke_at": state.last_spoke_at,
            "last_turn_ms": state.last_turn_ms,
            "recent": state.recent_segments,
            "transcript": state.recent_transcript(),
            "behavior": behavior.snapshot().to_dict(),
            "initiative": initiative_payload(),
            "voice": voice_store.get(),
            "ambient": _query_tap_ambient(),
            "campaign": campaigns.active().to_public_dict(),
            "campaigns": [p.to_summary() for p in campaigns.list_profiles()],
            "speakers": speaker_fn() if callable(speaker_fn) else {
                "enabled": False,
                "enrolled": [],
                "last_speaker": None,
                "last_confidence": None,
                "capturing": False,
            },
            "robot": robot_fn() if callable(robot_fn) else {
                "backend_connected": None,
                "host": None,
                "last_checked_at": None,
                "detail": "not checked",
                "disabled": False,
            },
        }

    def _publish_state(*_args: Any) -> None:
        try:
            event_bus.publish("state", _state_payload())
        except Exception:
            pass

    behavior_tool.bind(behavior, on_change=_publish_state)

    @app.get("/api/state")
    def api_state():
        return jsonify(_state_payload())

    @app.post("/api/mute")
    def api_mute():
        state.muted = True
        state.end_conversation()
        state.set_cue(ASLEEP)
        _publish_state()
        return jsonify(ok=True)

    @app.post("/api/unmute")
    def api_unmute():
        state.muted = False
        state.logging_enabled = True
        state.disarm_speak_now()
        if not state.asleep:
            state.set_cue(AWAKE)
        _publish_state()
        return jsonify(ok=True)

    @app.post("/api/mute_and_stop_logging")
    def api_mute_stop():
        state.muted = True
        state.logging_enabled = False
        state.end_conversation()
        state.set_cue(ASLEEP)
        return jsonify(ok=True)

    @app.post("/api/deep_night")
    def api_deep_night():
        body = request.get_json(force=True, silent=True)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return jsonify(error="body must be a JSON object"), 400
        if "asleep" in body:
            if not isinstance(body["asleep"], bool):
                return jsonify(error="asleep must be a boolean"), 400
            asleep = body["asleep"]
        else:
            asleep = not state.asleep
        if asleep != state.asleep:
            state.asleep = asleep
            if asleep:
                state.end_conversation()
                if not state.muted:
                    state.set_cue(ASLEEP)
            else:
                state.disarm_speak_now()
                if not state.muted:
                    state.set_cue(AWAKE)
        _publish_state()
        return jsonify(asleep=state.asleep, muted=state.muted)

    @app.post("/api/mode")
    def api_mode():
        body = request.get_json(force=True, silent=True)
        mode = str(body.get("mode", "")) if isinstance(body, dict) else ""
        if mode not in modes.MODES:
            return jsonify(error=f"mode must be one of {', '.join(modes.MODES)}"), 400
        state.mode = mode
        # Stamped with time.time(), the clock the responses route hands update_mode, or
        # a thinking mode set here would never time out.
        state.mode_set_at = time.time() if mode == modes.THINKING else None
        _publish_state()
        return jsonify(mode=mode)

    @app.post("/api/speak_now")
    def api_speak_now():
        if state.muted or state.asleep:
            return jsonify(ok=False, error="cannot arm speak_now while muted or in deep night"), 409
        state.arm_speak_now()
        _publish_state()
        return jsonify(ok=True)

    @app.post("/api/note")
    def api_note():
        return jsonify(result=note_tool.run(request.get_json(force=True) or {}))

    @app.post("/api/reindex")
    def api_reindex():
        prof = campaigns.active()
        return jsonify(indexing.index_notes(store, prof.notes_dir))

    def _end_session() -> Path:
        """Close the active session and queue it for summarising. The live indexer
        finishes first, so none of its writes can land after the summary replaces them."""
        live.end()
        path = log.end()
        state.session_started_at = None
        learner.queue([path])
        return path

    @app.post("/api/session/start")
    def api_session_start():
        prof = campaigns.active()
        # Notes edited between sessions are picked up here, so nobody has to restart
        # the stack to make the familiar aware of a new NPC.
        if prof.notes_dir.exists():
            indexing.index_notes(store, prof.notes_dir)
        # A start with a session still open used to start a new one over it, and the old
        # one was never summarised. It is ended and queued instead.
        ended = _end_session().stem if log.active_id is not None else None
        sid = log.start()
        state.session_started_at = time.time()
        state.clear_transcript()
        live.begin(log.path)
        # Queued after the start, so the learner sees the session and waits for its end.
        learner.catch_up()
        return jsonify(session=sid, ended=ended)

    @app.post("/api/session/end")
    def api_session_end():
        if log.active_id is None:  # SessionLog.path would raise, which Flask turns into a 500
            return jsonify(error="no active session"), 409
        path = _end_session()
        learner.kick()
        return jsonify(path=str(path), summarising=True)

    @app.post("/api/session/checkpoint")
    def api_session_checkpoint():
        if not checkpoint_lock.acquire(blocking=False):
            return jsonify(ok=False, error="checkpoint already in progress"), 409
        try:
            body = request.get_json(force=True, silent=True)
            if body is None:
                body = {}
            if not isinstance(body, dict):
                return jsonify(ok=False, error="body must be a JSON object"), 400
            if "force" in body:
                if not isinstance(body["force"], bool):
                    return jsonify(ok=False, error="force must be a boolean"), 400
                force = body["force"]
            else:
                force = False
            if log.active_id is None:
                return jsonify(ok=False, error="no active session"), 409
            if not state.muted and not state.asleep and not force:
                return jsonify(
                    ok=False,
                    error="table is unmuted; mute, enter deep night, or pass force=true",
                ), 409
            sid = log.active_id
            try:
                raw_path = log.path
                snapshot = log.snapshot_text()
            except RuntimeError:
                return jsonify(ok=False, error="no active session"), 409
            stop_fn = (
                (lambda: log.active_id != sid)
                if force
                else (lambda: log.active_id != sid or (not state.muted and not state.asleep))
            )
            players = load_players(campaigns.active().players_path)
            try:
                ckpt_path = checkpoint_session(
                    raw_path, store=store, llm=llm, players=players, stop=stop_fn, snapshot=snapshot
                )
            except (Paused, Purged) as e:
                return jsonify(ok=False, error=str(e)), 409
            except Exception as e:
                app.logger.error("session checkpoint failed: %s", e)
                return jsonify(ok=False, error=str(e)), 502
            if ckpt_path is None:
                return jsonify(ok=True, session=sid, summary_path=None, campaign_updated=False)
            try:
                merged = refresh_campaign(
                    campaigns.active().sessions_dir, store=store, llm=llm, players=players,
                    stop=stop_fn, include_checkpoints=True,
                )
            except (Paused, Purged) as e:
                return jsonify(ok=False, error=str(e)), 409
            except Exception as e:
                app.logger.warning("checkpoint campaign merge failed, leaving checkpoint summary: %s", e)
                merged = 0
            return jsonify(
                ok=True,
                session=sid,
                summary_path=str(ckpt_path),
                campaign_updated=merged > 0,
            )
        finally:
            checkpoint_lock.release()

    @app.post("/api/learn")
    def api_learn():
        return jsonify(queued=learner.catch_up(), paused=log.active_id is not None)

    @app.delete("/api/session/<sid>")
    def api_session_purge(sid: str):
        try:
            validate_session_id(sid)
        except ValueError:
            return jsonify(error="invalid session id"), 400
        # The live indexer would write the session's chunks back on its next batch, and
        # the session log would recreate the file on the next line.
        if sid == log.active_id:
            return jsonify(error="that session is in progress; end it first"), 409
        purged = purge_session(campaigns.active().sessions_dir, sid, store=store)
        # The campaign memory went with it if it held the session; the learner rebuilds it
        # from the summaries that remain, now if idle, else when the session ends.
        learner.kick()
        return jsonify(purged=purged)

    @app.get("/api/sessions")
    def api_sessions():
        sdir = campaigns.active().sessions_dir
        # Sessions only: the campaign memory and its .prev live in the same folder.
        ids = sorted({p.name.split(".", 1)[0] for pat in ("*.jsonl", "*.summary.md")
                      for p in sdir.glob(pat)})
        return jsonify(sessions=[{"id": i,
                                  "summary": (sdir / f"{i}.summary.md").exists(),
                                  "raw": (sdir / f"{i}.jsonl").exists()} for i in ids])

    @app.get("/api/speakers")
    def api_speakers():
        tap_role, tap_conf = _query_tap_current()
        if tap_role is not None:
            state.last_speaker = tap_role
            state.last_speaker_confidence = tap_conf
        return jsonify(
            enabled=bool(cfg.speakers.enabled),
            enrolled=speaker_registry.enrolled_roles(),
            last_speaker=state.last_speaker,
            last_confidence=state.last_speaker_confidence,
            capturing=_is_capturing(),
            backend=speaker_registry.embedder.name,
        )

    @app.post("/api/speakers/enroll")
    def api_speakers_enroll():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify(ok=False, error="body must be a JSON object"), 400
        role = str(body.get("role") or "").strip()
        if not role:
            return jsonify(ok=False, error="role is required"), 400
        players = load_players(campaigns.active().players_path)
        if role.lower() in {p.strip().lower() for p in players if p.strip()}:
            return jsonify(ok=False, error="use character or role names only, never real player names"), 400

        try:
            sample_rate = int(body.get("sample_rate") or 16000)
        except (TypeError, ValueError):
            return jsonify(ok=False, error="sample_rate must be an integer"), 400
        if sample_rate <= 0:
            return jsonify(ok=False, error="sample_rate must be positive"), 400

        audio: bytes | np.ndarray
        if "samples" in body and body["samples"] is not None:
            raw_samples = body["samples"]
            if not isinstance(raw_samples, list):
                return jsonify(ok=False, error="samples must be a JSON array"), 400
            try:
                audio = np.asarray(raw_samples, dtype=np.float32)
            except (TypeError, ValueError):
                return jsonify(ok=False, error="samples must be numeric"), 400
        elif "audio_b64" in body and body["audio_b64"] is not None:
            raw_b64 = body["audio_b64"]
            if not isinstance(raw_b64, str):
                return jsonify(ok=False, error="audio_b64 must be a string"), 400
            try:
                audio = base64.b64decode(raw_b64, validate=True)
            except (binascii.Error, ValueError):
                return jsonify(ok=False, error="invalid base64 audio_b64"), 400
        else:
            if not cfg.speakers.enabled:
                return jsonify(ok=False, error="speaker recognition is disabled"), 503
            try:
                duration_s = float(body.get("duration_s") or 5.0)
            except (TypeError, ValueError):
                return jsonify(ok=False, error="duration_s must be numeric"), 400
            duration_s = max(0.2, min(30.0, duration_s))
            with speaker_lock:
                if speaker_state["capture_count"] > 0:
                    return jsonify(ok=False, error="speaker capture already in progress"), 409
                speaker_state["capture_count"] += 1
            try:
                resp = tap_http.post(
                    f"{cfg.speakers.tap_url.rstrip('/')}/_tap/capture",
                    json={"duration_s": duration_s},
                    timeout=duration_s + 5.0,
                )
            except requests.RequestException:
                return jsonify(ok=False, error="audio tap unavailable"), 503
            finally:
                with speaker_lock:
                    speaker_state["capture_count"] = max(0, speaker_state["capture_count"] - 1)
            if resp.status_code == 408:
                return jsonify(ok=False, error="no speech captured from audio tap"), 400
            if resp.status_code == 409:
                return jsonify(ok=False, error="speaker capture already in progress"), 409
            if not resp.ok:
                return jsonify(ok=False, error=f"audio tap capture failed ({resp.status_code})"), 503
            cap_data = resp.json()
            try:
                audio = base64.b64decode(str(cap_data.get("audio_b64") or ""), validate=True)
                sample_rate = int(cap_data.get("sample_rate") or 16000)
            except Exception:
                return jsonify(ok=False, error="invalid audio returned from tap"), 503

        try:
            speaker_registry.enroll(role, audio, sample_rate=sample_rate, players=players)
        except (InsufficientSpeechError, ValueError) as exc:
            return jsonify(ok=False, error=str(exc)), 400
        _notify_tap_reload()
        return jsonify(ok=True, role=role, enrolled=speaker_registry.enrolled_roles())

    @app.delete("/api/speakers")
    def api_speakers_clear():
        speaker_registry.clear()
        state.last_speaker = None
        state.last_speaker_confidence = None
        _notify_tap_reload()
        return jsonify(ok=True, enrolled=[])

    @app.delete("/api/speakers/<role>")
    def api_speakers_delete(role: str):
        removed = speaker_registry.remove(role)
        if state.last_speaker and state.last_speaker.lower() == role.strip().lower():
            state.last_speaker = None
            state.last_speaker_confidence = None
        _notify_tap_reload()
        return jsonify(ok=True, removed=removed, enrolled=speaker_registry.enrolled_roles())

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/robot/status")
    def api_robot_status():
        probe_hook = app.extensions.get("robot_probe")
        if callable(probe_hook):
            res = probe_hook() or {}
            connected = bool(res.get("backend_connected"))
            with robot_lock:
                robot_state.update({
                    "backend_connected": connected,
                    "host": res.get("host", robot_state.get("host")),
                    "last_checked_at": res.get("last_checked_at", time.time()),
                    "detail": str(res.get("detail") or res.get("error") or ("connected" if connected else "offline")),
                    "disabled": bool(res.get("disabled", False)),
                })
                return jsonify(dict(robot_state))

        req_host = (request.args.get("host") or "").strip() or None
        want_probe = (request.args.get("probe") or "").strip() == "1"
        no_robot = _no_robot_env()
        with robot_lock:
            known_host = robot_state.get("host")
            if robot_state.get("mode") == "native" or _native_robot_env():
                robot_state["mode"] = "native"
                robot_state["disabled"] = False
                return jsonify(dict(robot_state))
        if _is_loopback_host(req_host) or (not no_robot and (want_probe or known_host)):
            rc = _load_robot_connect()
            if rc is not None:
                try:
                    daemon_port = int(request.args.get("daemon_port") or 8000)
                    rpc_port = int(request.args.get("rpc_port") or 7860)
                except (TypeError, ValueError):
                    return jsonify(error="daemon_port and rpc_port must be integers"), 400
                target_host = req_host or known_host
                res = rc.check_robot_status(
                    target_host,
                    session=robot_http,
                    daemon_port=daemon_port,
                    rpc_port=rpc_port,
                    fallback_host=None if target_host else "10.0.0.100",
                    allow_fallback=False,
                    timeout=2.5,
                )
                connected = bool(res.get("backend_connected"))
                with robot_lock:
                    robot_state.update({
                        "backend_connected": connected,
                        "host": res.get("host") or target_host,
                        "last_checked_at": time.time(),
                        "detail": str(res.get("detail") or res.get("error") or ("connected" if connected else "offline")),
                        "disabled": False,
                    })
                    return jsonify(dict(robot_state))
        if no_robot and not _is_loopback_host(known_host):
            with robot_lock:
                robot_state.update({
                    "backend_connected": False,
                    "host": None,
                    "last_checked_at": robot_state.get("last_checked_at"),
                    "detail": "robot disabled (FAMILIAR_NO_ROBOT=1)",
                    "disabled": True,
                })
                return jsonify(dict(robot_state))
        return jsonify(_robot_status())

    @app.post("/api/robot/reconnect")
    def api_robot_reconnect():
        body = request.get_json(force=True, silent=True)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return jsonify(ok=False, error="body must be a JSON object"), 400

        req_host_raw = body.get("host")
        req_host = str(req_host_raw).strip() if req_host_raw is not None else None
        if req_host == "":
            req_host = None

        reconnect_hook = app.extensions.get("robot_reconnect")
        if callable(reconnect_hook):
            res = reconnect_hook(body) or {}
            connected = bool(res.get("backend_connected"))
            ok = bool(res.get("ok", connected)) and connected
            detail = str(res.get("detail") or res.get("error") or ("connected" if ok else "unreachable"))
            with robot_lock:
                robot_state.update({
                    "backend_connected": connected,
                    "host": res.get("host", req_host),
                    "last_checked_at": time.time(),
                    "detail": detail,
                    "disabled": bool(res.get("disabled", False)),
                })
                snap = dict(robot_state)
            return jsonify(ok=ok, **snap), (200 if ok else 503)

        with robot_lock:
            if robot_state.get("mode") == "native" or _native_robot_env():
                robot_state["mode"] = "native"
                robot_state["disabled"] = False
                if not robot_state.get("backend_connected"):
                    robot_state["detail"] = "managed by reachy_familiar; waiting for audio stream"
                snap = dict(robot_state)
                return jsonify(ok=True, **snap), 200

        no_robot = _no_robot_env()
        if no_robot and not _is_loopback_host(req_host):
            with robot_lock:
                robot_state.update({
                    "backend_connected": False,
                    "host": None,
                    "last_checked_at": time.time(),
                    "detail": "robot disabled (FAMILIAR_NO_ROBOT=1)",
                    "disabled": True,
                })
                snap = dict(robot_state)
            return jsonify(ok=False, **snap), 503

        rc = _load_robot_connect()
        if rc is None:
            with robot_lock:
                robot_state.update({
                    "backend_connected": False,
                    "host": req_host,
                    "last_checked_at": time.time(),
                    "detail": "scripts/robot_connect.py unavailable",
                    "disabled": False,
                })
                snap = dict(robot_state)
            return jsonify(ok=False, **snap), 503

        try:
            daemon_port = int(body.get("daemon_port") or 8000)
            rpc_port = int(body.get("rpc_port") or 7860)
            port = int(body.get("port") or 8765)
            default_t = 2.5 if _is_loopback_host(req_host) else 15.0
            default_vt = 2.5 if _is_loopback_host(req_host) else 10.0
            timeout_s = float(body.get("timeout_s") or default_t)
            verify_timeout_s = float(body.get("verify_timeout_s") or default_vt)
        except (TypeError, ValueError):
            return jsonify(ok=False, error="ports and timeouts must be numeric"), 400

        host_ip = str(body.get("host_ip")).strip() if body.get("host_ip") else None
        voice = str(body.get("voice") or persona.tts_speaker or "bm_fable")
        fallback_host = None if req_host else (body.get("fallback_host") or "10.0.0.100")

        res = rc.connect_and_configure(
            req_host,
            host_ip=host_ip,
            port=port,
            voice=voice,
            daemon_port=daemon_port,
            rpc_port=rpc_port,
            fallback_host=fallback_host,
            allow_fallback=False,
            timeout_s=timeout_s,
            verify_timeout_s=verify_timeout_s,
            session=robot_http,
        )
        connected = bool(res.get("backend_connected"))
        ok = bool(res.get("ok", connected)) and connected
        detail = res.get("detail") or res.get("error") or (
            f"connected to {res.get('host')}" if connected else "robot unreachable"
        )
        with robot_lock:
            robot_state.update({
                "backend_connected": connected,
                "host": res.get("host") or req_host,
                "last_checked_at": time.time(),
                "detail": str(detail),
                "disabled": False if connected else no_robot,
            })
            snap = dict(robot_state)
        return jsonify(ok=ok, **snap), (200 if ok else 503)

    @app.post("/api/warmup")
    def api_warmup():
        body = request.get_json(force=True, silent=True)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return jsonify(ok=False, error="body must be a JSON object"), 400
        if "llm" in body:
            if not isinstance(body["llm"], bool):
                return jsonify(ok=False, error="llm must be a boolean"), 400
            want_llm = body["llm"]
        else:
            want_llm = os.environ.get("FAMILIAR_WARMUP_LLM") == "1"
        res = warmup(warm_llm=want_llm)
        return jsonify(ok=True, **res)

    def _on_robot_heartbeat(body: dict[str, Any]) -> dict[str, Any]:
        connected = bool(body.get("backend_connected", True))
        host = str(body.get("host") or "").strip() or robot_state.get("host")
        app_name = str(body.get("app") or "reachy_familiar").strip()
        detail = str(body.get("detail") or (f"connected via {app_name}" if connected else "disconnected"))
        mic_level_raw = body.get("mic_level")
        mic_level = float(mic_level_raw) if isinstance(mic_level_raw, (int, float)) else robot_state.get("mic_level", 0.0)
        with robot_lock:
            robot_state.update({
                "backend_connected": connected,
                "host": host,
                "last_checked_at": time.time(),
                "detail": detail,
                "disabled": False,
                "mode": "native",
                "mic_level": max(0.0, min(1.0, mic_level)),
            })
            return dict(robot_state)

    def _on_campaign_switched(prof: Any) -> None:
        """Isolate campaign memory, notes, sessions, and speaker profiles on team switch.

        Swaps filesystem pointers and clears old campaign vectors synchronously (<20 ms),
        then re-embeds the target campaign's notes & memory in a background thread so
        switching back to a large campaign never blocks or times out the HTTP UI request.
        """
        if log.active_id is not None:
            _end_session()
        prof.notes_dir.mkdir(parents=True, exist_ok=True)
        prof.sessions_dir.mkdir(parents=True, exist_ok=True)
        log.dir = prof.sessions_dir
        learner.dir = prof.sessions_dir
        learner.players_path = prof.players_path
        state.clear_transcript()
        state.last_speaker = None
        state.last_speaker_confidence = None
        # Keep shared D&D 2024/2014 `rules` corpus intact; swap only campaign-specific corpora.
        for corpus in ("notes", "homebrew", "sessions"):
            store.delete_corpus(corpus)
        speaker_registry.store_path = prof.speakers_path
        speaker_registry.load()
        _notify_tap_reload(store_path=prof.speakers_path)

        target_id = prof.id

        def _reindex_bg() -> None:
            try:
                if campaigns.active().id != target_id:
                    return
                if prof.notes_dir.exists():
                    indexing.index_notes(store, prof.notes_dir)
                mem_text = read_body(prof.sessions_dir / CAMPAIGN_FILE)
                if mem_text.strip() and campaigns.active().id == target_id:
                    save_manual_campaign(prof.sessions_dir, mem_text, store=store)
            except Exception as exc:
                app.logger.warning("background campaign reindex failed for %s: %s", target_id, exc)

        threading.Thread(target=_reindex_bg, daemon=True, name=f"reindex-{target_id}").start()

    def _on_chat_log(user_text: str, _reply: str) -> None:
        _log(user_text, Decision.ADDRESSED, spoke=False, speaker=state.last_speaker,
             speaker_confidence=state.last_speaker_confidence)

    register_companion_routes(
        app,
        behavior=behavior,
        voice_store=voice_store,
        event_bus=event_bus,
        state_payload=_state_payload,
        ambient_provider=_query_tap_ambient,
        sessions_dir=lambda: campaigns.active().sessions_dir,
        store=store,
        upstream=upstream,
        build_instructions=lambda q: _instructions(q, TurnTimer()),
        brain_tools=tools.specs,
        on_robot_heartbeat=_on_robot_heartbeat,
        on_chat_log=_on_chat_log,
        campaigns=campaigns,
        on_campaign_switched=_on_campaign_switched,
    )

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/brain.toml"
    app = create_app(cfg_path)
    cfg = load_config(cfg_path)
    warmup_fn = app.extensions.get("warmup")
    if callable(warmup_fn):
        warmup_fn(warm_llm=(os.environ.get("FAMILIAR_WARMUP_LLM") == "1"))
    try:
        app.run(host=cfg.server.host, port=cfg.server.port, threaded=True)
    finally:
        close_fn = app.extensions.get("close")
        if callable(close_fn):
            close_fn()
