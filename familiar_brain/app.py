from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context

from familiar_brain import indexing, tools
from familiar_brain.config import load_config
from familiar_brain.gate import Decision, Gate
from familiar_brain.knowledge import build_context
from familiar_brain.lexicon import Lexicon
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
from familiar_brain.tools import (
    initiative as initiative_tool,
    lookup_rule as lookup_tool,
    note as note_tool,
    recall as recall_tool,
    roll as roll_tool,
)
from familiar_brain.upstream import Upstream


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
                    classify_timeout_s=cfg.upstream.classify_timeout_s)
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
    upstream = Upstream(base_url=cfg.upstream.base_url, generate_slot=cfg.upstream.generate_slot,
                        brain_tool_names={s["name"] for s in tools.specs()})

    app = Flask(__name__)
    app.extensions["brain_state"] = state  # exposed for tests and the companion UI

    def _log(text: str, decision: Decision, spoke: bool) -> None:
        if state.logging_enabled and log.active_id is not None:
            log.append(text=text, decision=decision.value, spoke=spoke)

    def _instructions(text: str, timer: TurnTimer) -> str:
        """Retrieve for this utterance and prepend the result to the system prompt.

        Only ever called on an ADDRESSED turn, so an ignored line costs no retrieval.
        """
        try:
            context = build_context(store, text, chunks_per_corpus=cfg.knowledge.chunks_per_corpus,
                                    max_chars=cfg.knowledge.max_context_chars)
        except Exception as e:  # a broken store must not cost the table a reply
            app.logger.warning("retrieval failed: %s", e)
            context = ""
        timer.mark("retrieval")
        return render_system_prompt(persona, context)

    def _without_streaming(req: dict, text: str, decision: Decision, timer: TurnTimer, model: str):
        """The client calls /v1/responses without `stream` for its warmup probe and for
        history compaction. Both expect a JSON Response body, not an event stream."""
        if decision is not Decision.ADDRESSED:
            _log(text, decision, spoke=False)
            timer.finish(decision=decision.value, spoke=False)
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
        _log(text, decision, spoke=spoke)
        timer.mark("generation")
        timer.finish(decision=decision.value, spoke=spoke)
        return jsonify(body)

    @app.post("/v1/responses")
    def responses():
        req = request.get_json(force=True) or {}
        timer = TurnTimer()
        text = _correct_user_text(req, lexicon)
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
        outgoing = with_instructions_and_tools(req, instructions=_instructions(text, timer),
                                              extra_tools=tools.specs())

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
        return jsonify(path=str(log.end()))

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/brain.toml"
    app = create_app(cfg_path)
    cfg = load_config(cfg_path)
    app.run(host=cfg.server.host, port=cfg.server.port, threaded=True)
