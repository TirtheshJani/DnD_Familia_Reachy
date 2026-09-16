from __future__ import annotations

import logging
import sys
import time
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
    # asleep and awake are the mute and unmute cues, played by Choreographer.cue. Offered
    # to the mood call they would let an ordinary reply go out looking like a mute.
    reply_moods = tuple(m for m in persona.gestures if m not in (ASLEEP, AWAKE))
    choreo = Choreographer(
        persona, min_gap_s=cfg.motion.min_gesture_gap_s, reaction_gap_s=cfg.motion.reaction_gap_s,
        clock=time.time,
        mood_classifier=lambda t: llm.mood(t, moods=reply_moods),
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
        outgoing = with_instructions_and_tools(req, instructions=_instructions(text, timer),
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
            if decision is Decision.ADDRESSED:
                state.set_cue(AWAKE)
            timer.mark("gate")
        else:
            text = _correct_user_text(req, lexicon)
            state.remember(text)
            decision = gate.decide(state.recent_segments)
            timer.mark("gate")
            modes.update_mode(state, text, addressed=decision is Decision.ADDRESSED,
                              now=now, timeout_s=cfg.motion.mode_timeout_s)
        # SIGHT and GREETING in the timing line, so gate-report, which scores ADDRESSED
        # lines only, never counts a camera round trip as a reply to the table.
        label = decision.value if kind is None else kind.upper()
        if not wants_stream(req):
            return _without_streaming(req, text, decision, timer, model, kind)
        preface = _preface(text, decision, now)
        # How long the mood or reaction call took, which D2 and D9 say to watch live.
        timer.mark("preface")
        if decision is not Decision.ADDRESSED:
            if kind is None:
                _log(text, decision, spoke=False)
            timer.finish(decision=label, spoke=False)
            body = "".join(sse(e) for e in tool_call_events(preface or idle_items()))
            return Response(body, content_type="text/event-stream")
        outgoing = with_instructions_and_tools(req, instructions=_instructions(text, timer),
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
                for frame in upstream.stream(outgoing):
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
        state.muted = True; state.set_cue(ASLEEP); return jsonify(ok=True)

    @app.post("/api/unmute")
    def api_unmute():
        state.muted = False; state.logging_enabled = True; state.set_cue(AWAKE); return jsonify(ok=True)

    @app.post("/api/mute_and_stop_logging")
    def api_mute_stop():
        state.muted = True; state.logging_enabled = False; state.set_cue(ASLEEP); return jsonify(ok=True)

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
