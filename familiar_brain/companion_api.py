"""Phase 2 Companion UI API routes, SSE broadcaster, voice switcher, and silent chat.

Kept separate from familiar_brain/app.py so the core /v1/responses proxy stays focused
while the table-night control surface (TARS dials, camera consent, initiative, dice,
memory viewer/editor, silent chat, Kokoro voice picker, ambient mood, SSE push) lives
in one cohesive module.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, jsonify, request, stream_with_context

from familiar_brain.behavior import PRESETS, BehaviorSettings
from familiar_brain.campaign import CAMPAIGN_FILE, read_body, save_manual_campaign
from familiar_brain.responses_api import with_instructions_and_tools
from familiar_brain.tools import initiative as initiative_tool
from familiar_brain.tools import roll as roll_tool

KOKORO_VOICES: tuple[dict[str, str], ...] = (
    {"id": "bm_fable", "label": "Fable (British, crisp & dry)"},
    {"id": "bm_george", "label": "George (British, warm baritone)"},
    {"id": "bm_lewis", "label": "Lewis (British, thoughtful)"},
    {"id": "bf_emma", "label": "Emma (British, articulate)"},
    {"id": "am_fenrir", "label": "Fenrir (American, deep construct)"},
    {"id": "am_michael", "label": "Michael (American, clear)"},
    {"id": "af_heart", "label": "Heart (American, expressive)"},
)
_KOKORO_VOICE_RE = re.compile(r"^[abefhijpz][fm]_[a-z][a-z0-9_]*$")


class EventBus:
    """Thread-safe fan-out broadcaster for Server-Sent Events (/api/events)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: set[queue.Queue[tuple[str, dict[str, Any]]]] = set()

    def subscribe(self) -> queue.Queue[tuple[str, dict[str, Any]]]:
        q: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(maxsize=32)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[tuple[str, dict[str, Any]]]) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait((event, payload))
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait((event, payload))
                except Exception:
                    pass


class VoiceStore:
    """Persists the active Kokoro voice in data/voice.json for live pickup by run_s2s.py."""

    def __init__(self, path: Path | str, default_voice: str = "bm_fable") -> None:
        self.path = Path(path)
        self.default_voice = default_voice if _KOKORO_VOICE_RE.match(default_voice) else "bm_fable"
        self._lock = threading.Lock()
        self._voice = self.default_voice
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            v = raw.get("voice") if isinstance(raw, dict) else None
            if isinstance(v, str) and _KOKORO_VOICE_RE.match(v.strip()):
                self._voice = v.strip()
        except Exception:
            pass

    def get(self) -> str:
        with self._lock:
            return self._voice

    def set(self, voice: str) -> str:
        clean = str(voice).strip()
        if not _KOKORO_VOICE_RE.match(clean):
            raise ValueError(f"invalid Kokoro voice id: {voice!r}")
        with self._lock:
            self._voice = clean
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp")
            tmp.write_text(json.dumps({"voice": clean, "updated_at": time.time()}) + "\n", encoding="utf-8")
            try:
                os.replace(tmp, self.path)
            except PermissionError:
                time.sleep(0.02)
                os.replace(tmp, self.path)
            return self._voice


def initiative_payload() -> dict[str, Any]:
    rows = [{"name": n, "value": v} for n, v in initiative_tool.TRACKER.show()]
    return {
        "order": rows,
        "current": initiative_tool.TRACKER.current(),
    }


def register_companion_routes(
    app: Flask,
    *,
    behavior: BehaviorSettings,
    voice_store: VoiceStore,
    event_bus: EventBus,
    state_payload: Callable[[], dict[str, Any]],
    ambient_provider: Callable[[], dict[str, Any]],
    sessions_dir: Path | Callable[[], Path],
    store: Any,
    upstream: Any,
    build_instructions: Callable[[str], str],
    brain_tools: Callable[[], list[dict[str, Any]]],
    on_robot_heartbeat: Callable[[dict[str, Any]], dict[str, Any]],
    on_chat_log: Callable[[str, str], None] | None = None,
    campaigns: Any | None = None,
    on_campaign_switched: Callable[[Any], None] | None = None,
) -> None:
    """Register all Phase 2 table-control, TARS dial, memory, voice, chat, campaign, and SSE routes."""

    def _active_sessions_dir() -> Path:
        return sessions_dir() if callable(sessions_dir) else Path(sessions_dir)

    def _emit_state() -> None:
        try:
            event_bus.publish("state", state_payload())
        except Exception:
            pass

    @app.get("/api/events")
    def api_events():
        one_shot = (request.args.get("once") or "").strip() == "1"

        def stream():
            init_data = json.dumps(state_payload(), ensure_ascii=False)
            yield f"event: state\ndata: {init_data}\n\n"
            if one_shot:
                return
            q = event_bus.subscribe()
            try:
                while True:
                    try:
                        ev, payload = q.get(timeout=15.0)
                        data = json.dumps(payload, ensure_ascii=False)
                        yield f"event: {ev}\ndata: {data}\n\n"
                    except queue.Empty:
                        yield ": ping\n\n"
            finally:
                event_bus.unsubscribe(q)

        return Response(
            stream_with_context(stream()),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/behavior")
    def api_behavior_get():
        return jsonify(
            **behavior.snapshot().to_dict(),
            presets=list(PRESETS.keys()),
        )

    @app.post("/api/behavior")
    def api_behavior_post():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify(error="body must be a JSON object"), 400
        try:
            snap = behavior.update(**body)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        _emit_state()
        return jsonify(ok=True, **snap.to_dict())

    @app.post("/api/camera")
    def api_camera():
        body = request.get_json(force=True, silent=True)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return jsonify(error="body must be a JSON object"), 400
        if "enabled" in body:
            if not isinstance(body["enabled"], bool):
                return jsonify(error="enabled must be a boolean"), 400
            enabled = body["enabled"]
        else:
            enabled = not behavior.snapshot().camera_enabled
        snap = behavior.update(camera_enabled=enabled)
        _emit_state()
        return jsonify(ok=True, camera_enabled=snap.camera_enabled)

    @app.get("/api/initiative")
    def api_initiative_get():
        return jsonify(initiative_payload())

    @app.post("/api/initiative")
    def api_initiative_post():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify(error="body must be a JSON object"), 400
        action = str(body.get("action", "")).strip().lower()
        if action == "add":
            name = str(body.get("name", "")).strip()
            if not name:
                return jsonify(error="name is required"), 400
            try:
                val = int(body["value"])
            except (KeyError, TypeError, ValueError):
                return jsonify(error="value must be an integer"), 400
            result = initiative_tool.run({"action": "add", "name": name, "value": val})
        elif action == "remove":
            name = str(body.get("name", "")).strip()
            if not name:
                return jsonify(error="name is required"), 400
            removed = initiative_tool.TRACKER.remove(name)
            result = f"Removed {name}." if removed else f"{name} was not in the order."
        elif action in ("next", "clear", "show"):
            result = initiative_tool.run({"action": action})
        else:
            return jsonify(error="action must be one of add, remove, next, clear, show"), 400
        _emit_state()
        return jsonify(ok=True, result=result, **initiative_payload())

    @app.post("/api/roll")
    def api_roll():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify(error="body must be a JSON object"), 400
        expr = str(body.get("expression", "")).strip()
        adv = str(body.get("advantage", "none")).strip().lower()
        if not expr:
            return jsonify(error="expression is required"), 400
        if adv not in ("none", "adv", "dis"):
            return jsonify(error="advantage must be none, adv, or dis"), 400
        result = roll_tool.run({"expression": expr, "advantage": adv})
        return jsonify(ok=True, expression=expr, advantage=adv, result=result)

    @app.get("/api/memory")
    def api_memory_get():
        sdir = _active_sessions_dir()
        mem_path = sdir / CAMPAIGN_FILE
        campaign_text = read_body(mem_path)
        summaries = []
        if sdir.exists():
            for p in sorted(sdir.glob("*.summary.md"), reverse=True)[:10]:
                sid = p.name[: -len(".summary.md")]
                try:
                    content = p.read_text(encoding="utf-8")
                except OSError:
                    content = ""
                summaries.append({"id": sid, "text": content})
        return jsonify(campaign_memory=campaign_text, summaries=summaries)

    @app.put("/api/memory")
    def api_memory_put():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("campaign_memory"), str):
            return jsonify(error="campaign_memory string is required"), 400
        try:
            chunks = save_manual_campaign(_active_sessions_dir(), body["campaign_memory"], store=store)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        _emit_state()
        return jsonify(ok=True, indexed_chunks=chunks)

    @app.post("/api/chat")
    def api_chat():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("text"), str) or not body["text"].strip():
            return jsonify(error="text is required"), 400
        text = body["text"].strip()
        from familiar_brain.tools import behavior_setting as behavior_tool

        voice_cmd = behavior_tool.parse_voice_setting_command(text)
        if voice_cmd is not None:
            behavior.update(**{voice_cmd[0]: voice_cmd[1]})
        instructions = build_instructions(text)
        req_payload = with_instructions_and_tools(
            {
                "model": "familiar",
                "stream": True,
                "input": [{"role": "user", "content": text}],
                "tools": [],
            },
            instructions=instructions,
            extra_tools=brain_tools(),
        )
        deltas: list[str] = []
        try:
            for frame in upstream.stream(req_payload, must_speak=True):
                for line in frame.splitlines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        ev = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "response.output_text.delta" and isinstance(ev.get("delta"), str):
                        deltas.append(ev["delta"])
        except Exception as exc:
            return jsonify(ok=False, error=f"upstream unavailable: {exc}"), 503
        reply = "".join(deltas).strip() or "(no reply)"
        if on_chat_log is not None:
            on_chat_log(text, reply)
        _emit_state()
        return jsonify(ok=True, reply=reply)

    @app.get("/api/voice")
    def api_voice_get():
        return jsonify(voice=voice_store.get(), voices=list(KOKORO_VOICES))

    @app.post("/api/voice")
    def api_voice_post():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("voice"), str):
            return jsonify(error="voice string is required"), 400
        try:
            selected = voice_store.set(body["voice"])
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        _emit_state()
        return jsonify(ok=True, voice=selected, voices=list(KOKORO_VOICES))

    @app.get("/api/ambient")
    def api_ambient_get():
        return jsonify(ambient_provider())

    @app.post("/api/robot/heartbeat")
    def api_robot_heartbeat():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify(error="body must be a JSON object"), 400
        snap = on_robot_heartbeat(body)
        _emit_state()
        return jsonify(ok=True, **snap)

    if campaigns is not None:
        @app.get("/api/campaigns")
        def api_campaigns_get():
            return jsonify(
                active=campaigns.active().to_public_dict(),
                profiles=[p.to_summary() for p in campaigns.list_profiles()],
            )

        @app.post("/api/campaigns")
        def api_campaigns_create():
            body = request.get_json(force=True, silent=True)
            if not isinstance(body, dict):
                return jsonify(error="body must be a JSON object"), 400
            raw_roster = body.get("roster")
            if isinstance(raw_roster, str):
                roster_list = [r.strip() for r in raw_roster.split(",") if r.strip()]
            elif isinstance(raw_roster, list):
                roster_list = [str(r).strip() for r in raw_roster if str(r).strip()]
            else:
                roster_list = None
            try:
                prof = campaigns.create_profile(
                    str(body.get("name") or ""),
                    setting=str(body.get("setting") or ""),
                    bound_to=str(body.get("bound_to") or ""),
                    roster=roster_list,
                    secrets=str(body.get("secrets") or ""),
                )
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
            _emit_state()
            return jsonify(
                ok=True,
                profile=prof.to_public_dict(),
                profiles=[p.to_summary() for p in campaigns.list_profiles()],
            )

        @app.post("/api/campaigns/switch")
        def api_campaigns_switch():
            body = request.get_json(force=True, silent=True)
            if not isinstance(body, dict) or not str(body.get("id") or "").strip():
                return jsonify(error="id is required"), 400
            try:
                prof = campaigns.switch(str(body["id"]))
            except ValueError as exc:
                return jsonify(error=str(exc)), 404
            if on_campaign_switched is not None:
                on_campaign_switched(prof)
            _emit_state()
            return jsonify(
                ok=True,
                active=prof.to_public_dict(),
                profiles=[p.to_summary() for p in campaigns.list_profiles()],
            )

        @app.delete("/api/campaigns/<cid>")
        def api_campaigns_delete(cid: str):
            try:
                removed = campaigns.delete_profile(cid)
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
            _emit_state()
            return jsonify(
                ok=True,
                removed=removed,
                profiles=[p.to_summary() for p in campaigns.list_profiles()],
            )

        @app.post("/api/roster")
        def api_roster_post():
            body = request.get_json(force=True, silent=True)
            if not isinstance(body, dict):
                return jsonify(error="body must be a JSON object"), 400
            action = str(body.get("action") or "").strip().lower()
            role = str(body.get("role") or "").strip()
            try:
                if action == "add":
                    roster = campaigns.add_roster_role(role)
                elif action == "remove":
                    roster = campaigns.remove_roster_role(role)
                else:
                    return jsonify(error="action must be 'add' or 'remove'"), 400
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
            _emit_state()
            return jsonify(ok=True, roster=roster)

        @app.post("/api/privacy-names")
        def api_privacy_names_post():
            body = request.get_json(force=True, silent=True)
            if not isinstance(body, dict):
                return jsonify(error="body must be a JSON object"), 400
            action = str(body.get("action") or "").strip().lower()
            name = str(body.get("name") or "").strip()
            try:
                if action == "add":
                    names = campaigns.add_privacy_name(name)
                elif action == "remove":
                    names = campaigns.remove_privacy_name(name)
                else:
                    return jsonify(error="action must be 'add' or 'remove'"), 400
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
            _emit_state()
            return jsonify(ok=True, privacy_names=names)
