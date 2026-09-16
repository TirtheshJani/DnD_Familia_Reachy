from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context

from familiar_brain import indexing, modes, tools
from familiar_brain.attention import Attention
from familiar_brain.config import load_config
from familiar_brain.gate import Decision, Gate
from familiar_brain.knowledge import build_context
from familiar_brain.lexicon import Lexicon
from familiar_brain.llm import REACTION_INSTRUCTION, LLMClient
from familiar_brain.motion import REACTION_MOODS, Choreographer
from familiar_brain.persona import load_persona, render_system_prompt
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
from familiar_brain.robot import RobotDaemon
from familiar_brain.session import SessionLog
from familiar_brain.summariser import (
    apply_retention,
    load_players,
    purge_session,
    summarise_in_background,
)
from familiar_brain.state import BrainState
from familiar_brain.timing import TurnTimer
from familiar_brain.tools import (
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
    glance_at: float
    gesture_at: float
    reaction_at: float
    cue: str | None


def _without_gestures(req: dict) -> dict:
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
    offered = req.get("tools")
    if isinstance(offered, list):
        req = {**req, "tools": [t for t in offered
                                if not (isinstance(t, dict) and t.get("name") in GESTURE_TOOLS)]}
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
                out.append({**part, "text": text}); folded = True
            elif not is_text:
                out.append(part)
        item["content"] = out if folded else content
        return text
    return text


def create_app(config_path: Path | str = "config/brain.toml") -> Flask:
    cfg = load_config(config_path)
    persona = load_persona(cfg.paths.persona)
    lexicon = Lexicon.load(cfg.paths.lexicon)
    state = BrainState()
    log = SessionLog(cfg.paths.sessions_dir)
    llm = LLMClient(base_url=cfg.upstream.base_url, classify_slot=cfg.upstream.classify_slot,
                    classify_timeout_s=cfg.upstream.classify_timeout_s,
                    generate_slot=cfg.upstream.generate_slot)
    gate = Gate(state=state, name=persona.name, aliases=persona.aliases,
                follow_up_window_s=cfg.gate.follow_up_window_s,
                max_follow_ups=cfg.gate.max_follow_ups,
                classifier=lambda segs: llm.classify(segs, persona_name=persona.name))
    store = indexing.open_store(cfg)
    notes_dir = Path(cfg.knowledge.notes_dir)
    if notes_dir.exists():
        indexing.index_notes(store, notes_dir)
    for mod in (roll_tool, initiative_tool, note_tool, lookup_tool, recall_tool):
        tools.register(mod)
    note_tool.bind(log)
    lookup_tool.bind(store)
    recall_tool.bind(store)
    sessions_dir = Path(cfg.paths.sessions_dir)
    apply_retention(sessions_dir, store=store, retention_days=cfg.knowledge.retention_days)
    players = load_players(cfg.paths.players)
    upstream = Upstream(base_url=cfg.upstream.base_url, generate_slot=cfg.upstream.generate_slot,
                        brain_tool_names={s["name"] for s in tools.specs()})
    robot = RobotDaemon(cfg.robot.daemon_url, timeout_s=cfg.robot.doa_timeout_s)
    # No mood call on addressed turns: it ran ahead of the first word, and live on
    # 2026-09-16 it timed out on 11 of 11 addressed turns under GPU contention, always
    # falling back to attentive, the emotion that played over Rocky's speech. The keyword
    # rules alone choose a reply's gesture (motion.for_addressed).
    choreo = Choreographer(
        persona, min_gap_s=cfg.motion.min_gesture_gap_s, reaction_gap_s=cfg.motion.reaction_gap_s,
        clock=time.time,
        mood_classifier=None,
        reaction_classifier=lambda t: llm.mood(t, moods=REACTION_MOODS + ("none",), prompt=REACTION_INSTRUCTION),
    )
    attention = Attention(state=state, doa=robot.doa, follow_up_window_s=cfg.gate.follow_up_window_s,
                          linger_s=cfg.motion.tracking_linger_s, glance_gap_s=cfg.motion.glance_gap_s)

    app = Flask(__name__)
    app.extensions["brain_state"] = state  # exposed for tests and the companion UI

    def _log(text: str, decision: Decision, spoke: bool) -> None:
        if state.logging_enabled and log.active_id is not None:
            log.append(text=text, decision=decision.value, spoke=spoke)

    def _instructions(text: str, timer: TurnTimer) -> str:
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
        return render_system_prompt(persona, context, mode=state.mode)

    def _preface(text: str, decision: Decision, now: float) -> list[dict]:
        """Robot-side tool calls that go out ahead of (or instead of) speech this turn.

        Order: attention first (a tracking toggle or a glance), then a pending cue (mute,
        unmute), then the mood of the reply on an addressed turn or a reaction to the
        table on any other. The tracking toggle leads so that on the turn after a mute
        the head stops following faces before the sleep gesture plays, not after it. A
        muted brain sends at most the tracking-off toggle and its cue: no glance, no
        reaction. A continuation passes no text, so it carries no mood or reaction.
        """
        items: list[dict] = []
        try:
            items += attention.items(decision, now)
            cue = state.consume_cue()
            if cue:
                items += choreo.cue(cue, now)
            if state.muted or not text:
                return items
            if decision is Decision.ADDRESSED:
                items += choreo.for_addressed(text, now)
            else:
                items += choreo.for_table(text, now)
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
        return _PrefaceStart(tracking=state.tracking, glance_at=state.last_glance_at,
                             gesture_at=choreo.last_at, reaction_at=choreo.last_reaction_at, cue=cue)

    def _rewind_if_revision(key: str | None, now: float) -> None:
        """Put back what a superseded version of this utterance spent, before its revision's
        preface runs.

        With smart_turn on, speech-to-speech asks for a reply at the player's first pause.
        If the player carries on inside its reopen window, it replaces that user message
        with a transcript of the longer audio, asks again, and drops everything the first
        response sent, gestures included. A preface spends state only once: a tracking
        toggle goes out only on a change, a cue only once, and a gesture or glance starts
        its gap. Without this the revision the robot does get carries none of them, so
        Rocky never tracks for the whole exchange, or keeps following faces after a mute.

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
            state.last_glance_at = previous.glance_at
            choreo.last_at = previous.gesture_at
            choreo.last_reaction_at = previous.reaction_at
            if previous.cue:
                state.set_cue(state.consume_cue() or previous.cue)
        last_utterance.update(key=key, at=now, start=_preface_start())

    def _without_streaming(req: dict, text: str, decision: Decision, timer: TurnTimer, model: str,
                           kind: str | None):
        """The client calls /v1/responses without `stream` for its warmup probe and for
        history compaction. Both expect a JSON Response body, not an event stream.

        No preface on this path: a JSON body is not relayed to the robot as it arrives.
        A continuation (`kind` not None) is answered but never logged, see responses().
        """
        label = decision.value if kind is None else kind.upper()
        if decision is not Decision.ADDRESSED:
            if kind is None:
                _log(text, decision, spoke=False)
            timer.finish(decision=label, spoke=False)
            return jsonify(idle_response_json(model=model))
        outgoing = with_instructions_and_tools(_without_gestures(req),
                                              instructions=_instructions(text, timer),
                                              extra_tools=tools.specs())
        try:
            body, spoke = upstream.complete(outgoing), True
        except Exception as e:  # upstream down: degrade to silence
            app.logger.error("upstream failed (non-streaming): %s", e)
            body, spoke = idle_response_json(model=model), False
        # Deliberately no mark_spoke here. This path serves the client's warmup probe
        # and history compaction, neither of which is speech at the table, and marking
        # it would open the follow-up window before anyone has said anything.
        if kind is None:
            _log(text, decision, spoke=spoke)
        timer.mark("generation")
        timer.finish(decision=label, spoke=spoke)
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
        if kind == "sight":
            # The frame the robot just took. Not an utterance: no lexicon, no log, no
            # follow-up accounting, and the frame is never written anywhere (D5). Muted
            # means eyes closed.
            text = ""
            decision = Decision.IGNORE if state.muted else Decision.ADDRESSED
            timer.mark("gate")
        elif kind == "greeting":
            # The app's session-open prompt. Answer it, but do not open the follow-up
            # window: nobody at the table has said anything yet (compare deviation E7).
            text = ""
            decision = Decision.IGNORE if state.muted else Decision.ADDRESSED
            timer.mark("gate")
        else:
            text = _correct_user_text(req, lexicon)
            state.remember(text)
            decision = gate.decide(state.recent_segments)
            timer.mark("gate")
            modes.update_mode(state, text, addressed=decision is Decision.ADDRESSED,
                              now=now, timeout_s=cfg.motion.mode_timeout_s)
        # SIGHT and GREETING in the timing line, so gate-report's latency figures, which
        # read ADDRESSED records only, never include a camera round trip or the greeting.
        label = decision.value if kind is None else kind.upper()
        if not wants_stream(req):
            return _without_streaming(req, text, decision, timer, model, kind)
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
                _log(text, decision, spoke=False)
            timer.finish(decision=label, spoke=False)
            body = "".join(sse(e) for e in tool_call_events(preface or idle_items()))
            return Response(body, content_type="text/event-stream")
        outgoing = with_instructions_and_tools(_without_gestures(req),
                                              instructions=_instructions(text, timer),
                                              extra_tools=tools.specs())

        def gen():
            spoke = False
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
                    _log(text, decision, spoke=spoke)
                timer.mark("generation")
                timer.finish(decision=label, spoke=spoke)

        return Response(stream_with_context(gen()), content_type="text/event-stream")

    @app.get("/api/state")
    def api_state():
        # Apply the thinking-mode timeout here as well, so the companion UI never shows a
        # mode the next turn would already have ended.
        modes.update_mode(state, "", addressed=False, now=time.time(),
                          timeout_s=cfg.motion.mode_timeout_s)
        return jsonify(muted=state.muted, mode=state.mode, tracking=state.tracking,
                       logging=state.logging_enabled, session=log.active_id,
                       recent=state.recent_segments, last_spoke_at=state.last_spoke_at)

    @app.post("/api/mute")
    def api_mute():
        state.muted = True; state.end_conversation(); state.set_cue(ASLEEP); return jsonify(ok=True)

    @app.post("/api/unmute")
    def api_unmute():
        state.muted = False; state.logging_enabled = True; state.set_cue(AWAKE); return jsonify(ok=True)

    @app.post("/api/mute_and_stop_logging")
    def api_mute_stop():
        state.muted = True; state.logging_enabled = False; state.end_conversation(); state.set_cue(ASLEEP)
        return jsonify(ok=True)

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
        return jsonify(mode=mode)

    @app.post("/api/speak_now")
    def api_speak_now():
        state.arm_speak_now(); return jsonify(ok=True)

    @app.post("/api/note")
    def api_note():
        return jsonify(result=note_tool.run(request.get_json(force=True) or {}))

    @app.post("/api/reindex")
    def api_reindex():
        return jsonify(indexing.index_notes(store, notes_dir))

    @app.post("/api/session/start")
    def api_session_start():
        # Notes edited between sessions are picked up here, so nobody has to restart
        # the stack to make the familiar aware of a new NPC.
        if notes_dir.exists():
            indexing.index_notes(store, notes_dir)
        return jsonify(session=log.start())

    @app.post("/api/session/end")
    def api_session_end():
        if log.active_id is None:  # SessionLog.path would raise, which Flask turns into a 500
            return jsonify(error="no active session"), 409
        path = log.end()
        summarise_in_background(path, store=store, llm=llm, players=players)
        return jsonify(path=str(path), summarising=True)

    @app.delete("/api/session/<sid>")
    def api_session_purge(sid: str):
        return jsonify(purged=purge_session(sessions_dir, sid, store=store))

    @app.get("/api/sessions")
    def api_sessions():
        ids = sorted({p.stem.replace(".summary", "") for p in sessions_dir.glob("*")})
        return jsonify(sessions=[{"id": i,
                                  "summary": (sessions_dir / f"{i}.summary.md").exists(),
                                  "raw": (sessions_dir / f"{i}.jsonl").exists()} for i in ids])

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/brain.toml"
    app = create_app(cfg_path)
    cfg = load_config(cfg_path)
    app.run(host=cfg.server.host, port=cfg.server.port, threaded=True)
