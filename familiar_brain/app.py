from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context

from familiar_brain import tools
from familiar_brain.config import load_config
from familiar_brain.gate import Decision, Gate
from familiar_brain.llm import LLMClient
from familiar_brain.persona import load_persona, render_system_prompt
from familiar_brain.responses_api import (
    idle_response_events,
    idle_response_json,
    latest_user_text,
    sse,
    wants_stream,
    with_instructions_and_tools,
)
from familiar_brain.session import SessionLog
from familiar_brain.state import BrainState
from familiar_brain.timing import TurnTimer
from familiar_brain.tools import initiative as initiative_tool, note as note_tool, roll as roll_tool
from familiar_brain.upstream import Upstream


def create_app(config_path: Path | str = "config/brain.toml") -> Flask:
    cfg = load_config(config_path)
    persona = load_persona(cfg.paths.persona)
    system_prompt = render_system_prompt(persona)
    state = BrainState()
    log = SessionLog(cfg.paths.sessions_dir)
    llm = LLMClient(base_url=cfg.upstream.base_url, classify_slot=cfg.upstream.classify_slot,
                    classify_timeout_s=cfg.upstream.classify_timeout_s)
    gate = Gate(state=state, name=persona.name, aliases=persona.aliases,
                follow_up_window_s=cfg.gate.follow_up_window_s,
                classifier=lambda segs: llm.classify(segs, persona_name=persona.name))
    for mod in (roll_tool, initiative_tool, note_tool):
        tools.register(mod)
    note_tool.bind(log)
    upstream = Upstream(base_url=cfg.upstream.base_url, generate_slot=cfg.upstream.generate_slot,
                        brain_tool_names={s["name"] for s in tools.specs()})

    app = Flask(__name__)
    app.extensions["brain_state"] = state  # exposed for tests and the companion UI

    def _log(text: str, decision: Decision, spoke: bool) -> None:
        if state.logging_enabled and log.active_id is not None:
            log.append(text=text, decision=decision.value, spoke=spoke)

    def _without_streaming(req: dict, text: str, decision: Decision, timer: TurnTimer, model: str):
        """The client calls /v1/responses without `stream` for its warmup probe and for
        history compaction. Both expect a JSON Response body, not an event stream."""
        if decision is not Decision.ADDRESSED:
            _log(text, decision, spoke=False)
            timer.finish(decision=decision.value, spoke=False)
            return jsonify(idle_response_json(model=model))
        outgoing = with_instructions_and_tools(req, instructions=system_prompt, extra_tools=tools.specs())
        try:
            body, spoke = upstream.complete(outgoing), True
        except Exception as e:  # upstream down: degrade to silence
            app.logger.error("upstream failed (non-streaming): %s", e)
            body, spoke = idle_response_json(model=model), False
        if spoke:
            state.mark_spoke()
        _log(text, decision, spoke=spoke)
        timer.mark("generation")
        timer.finish(decision=decision.value, spoke=spoke)
        return jsonify(body)

    @app.post("/v1/responses")
    def responses():
        req = request.get_json(force=True) or {}
        timer = TurnTimer()
        text = latest_user_text(req)
        state.remember(text)
        decision = gate.decide(state.recent_segments)
        timer.mark("gate")
        if not wants_stream(req):
            return _without_streaming(req, text, decision, timer, str(req.get("model") or "familiar"))
        if decision is not Decision.ADDRESSED:
            _log(text, decision, spoke=False)
            timer.finish(decision=decision.value, spoke=False)
            body = "".join(sse(e) for e in idle_response_events())
            return Response(body, content_type="text/event-stream")
        outgoing = with_instructions_and_tools(req, instructions=system_prompt, extra_tools=tools.specs())

        def gen():
            spoke = False
            try:
                for frame in upstream.stream(outgoing):
                    if "response.output_text.delta" in frame:
                        spoke = True
                    yield frame
            except Exception as e:  # upstream down: degrade to silence
                app.logger.error("upstream failed: %s", e)
                for ev in idle_response_events():
                    yield sse(ev)
            finally:
                if spoke:
                    state.mark_spoke()
                _log(text, decision, spoke=spoke)
                timer.mark("generation")
                timer.finish(decision=decision.value, spoke=spoke)

        return Response(stream_with_context(gen()), content_type="text/event-stream")

    @app.get("/api/state")
    def api_state():
        return jsonify(muted=state.muted, logging=state.logging_enabled, session=log.active_id,
                       recent=state.recent_segments, last_spoke_at=state.last_spoke_at)

    @app.post("/api/mute")
    def api_mute():
        state.muted = True; return jsonify(ok=True)

    @app.post("/api/unmute")
    def api_unmute():
        state.muted = False; state.logging_enabled = True; return jsonify(ok=True)

    @app.post("/api/mute_and_stop_logging")
    def api_mute_stop():
        state.muted = True; state.logging_enabled = False; return jsonify(ok=True)

    @app.post("/api/speak_now")
    def api_speak_now():
        state.arm_speak_now(); return jsonify(ok=True)

    @app.post("/api/note")
    def api_note():
        return jsonify(result=note_tool.run(request.get_json(force=True) or {}))

    @app.post("/api/session/start")
    def api_session_start():
        return jsonify(session=log.start())

    @app.post("/api/session/end")
    def api_session_end():
        if log.active_id is None:  # SessionLog.path would raise, which Flask turns into a 500
            return jsonify(error="no active session"), 409
        return jsonify(path=str(log.end()))

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/brain.toml"
    app = create_app(cfg_path)
    cfg = load_config(cfg_path)
    app.run(host=cfg.server.host, port=cfg.server.port, threaded=True)
