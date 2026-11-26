"""Entry point for the native Reachy Mini app `reachy_familiar`."""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Any

import requests

from reachy_familiar.audio_bridge import RealtimeAudioBridge
from reachy_familiar.creature import CreatureController
from reachy_familiar.discovery import (
    LaptopEndpoint,
    discover_laptop,
    poll_wake_status,
    resolve_brain_url,
    sleep_laptop,
    wake_laptop,
)
from reachy_familiar.web import CompanionProxyServer

log = logging.getLogger("reachy_familiar")

try:
    from reachy_mini import ReachyMini, ReachyMiniApp  # type: ignore[import-not-found]
except ImportError:  # Allows unit testing on dev laptops without the robot SDK installed.
    ReachyMini = Any  # type: ignore[misc,assignment]

    class ReachyMiniApp:  # type: ignore[no-redef]
        custom_app_url: str | None = None
        dont_start_webserver: bool = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.stop_event = threading.Event()

        def wrapped_run(self, *args: Any, **kwargs: Any) -> None:
            try:
                self.run(None, self.stop_event)
            except KeyboardInterrupt:
                self.stop_event.set()


class ReachyFamiliarApp(ReachyMiniApp):
    """Native Reachy Mini app that wakes Rocky's RTX 4080 brain and runs living-pet behaviors."""

    custom_app_url: str | None = "http://0.0.0.0:8043"
    dont_start_webserver: bool = True

    def __init__(
        self,
        *args: Any,
        web_port: int = 8043,
        daemon_url: str = "http://127.0.0.1:8000",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.web_port = web_port
        self.daemon_url = daemon_url
        self._state_lock = threading.Lock()
        self._boot_state: dict[str, Any] = {
            "ready": False,
            "brain_ready": False,
            "voice_ready": False,
            "state": "waking",
            "stage": "discovering",
            "detail": "Searching LAN for your Zephyrus RTX 4080 laptop...",
            "brain_url": None,
            "audio_connected": False,
            "mic_level": 0.0,
            "mic_alive": False,
            "mic_stalled": False,
            "speaker_boost_db": 6.0,
        }

    def _set_boot_state(self, **kwargs: Any) -> None:
        with self._state_lock:
            self._boot_state.update(kwargs)

    def boot_status(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._boot_state)

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

        # Trap SIGINT/SIGTERM cleanly so clicking Stop in Reachy Desktop App logs zero tracebacks.
        def _handle_stop_sig(_signum: int, _frame: Any) -> None:
            stop_event.set()

        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _handle_stop_sig)
                except ValueError:
                    pass

        creature = CreatureController(daemon_url=self.daemon_url, reachy_mini=reachy_mini)
        ep: LaptopEndpoint | None = None
        bridge_thread: threading.Thread | None = None
        bridge: RealtimeAudioBridge | None = None

        def _action_speak_test() -> dict[str, Any]:
            if bridge is not None:
                return bridge.play_test_tone()
            # Even while voice is still warming up, play the test chime directly on Reachy's speaker.
            tmp_bridge = RealtimeAudioBridge(
                host=ep.host if ep else "127.0.0.1",
                port=ep.ws_port if ep else 8765,
                reachy_mini=reachy_mini,
                creature=creature,
                speaker_boost_db=float(self.boot_status().get("speaker_boost_db", 6.0)),
            )
            return tmp_bridge.play_test_tone()

        def _action_reconnect_audio() -> dict[str, Any]:
            if bridge is None:
                return {"ok": False, "error": "Voice engine is still warming up; audio bridge not started yet."}
            bridge.request_reconnect()
            return {"ok": True, "detail": "Reconnecting Reachy realtime audio stream..."}

        web = CompanionProxyServer(
            port=self.web_port,
            status_provider=self.boot_status,
            action_handlers={
                "speak_test": _action_speak_test,
                "reconnect_audio": _action_reconnect_audio,
            },
        )
        web.start()

        try:
            # 1. Discover the laptop Wake Daemon on LAN (works on any Wi-Fi / hotspot).
            while not stop_event.is_set() and ep is None:
                ep = discover_laptop(timeout_s=3.0)
                if ep is None:
                    self._set_boot_state(
                        ready=False,
                        brain_ready=False,
                        voice_ready=False,
                        state="waking",
                        stage="discovering",
                        detail="Waiting for Zephyrus Wake Daemon on LAN (ensure laptop is on Wi-Fi)...",
                    )
                    stop_event.wait(2.0)

            if stop_event.is_set() or ep is None:
                return

            # 2. Wake the RTX 4080 stack; open Companion UI as soon as familiar-brain is up (~15s),
            # then start the audio bridge as soon as Parakeet STT & Kokoro TTS are ready.
            self._set_boot_state(
                stage="starting",
                detail=f"Waking RTX 4080 brain at {ep.host}...",
                brain_url=ep.brain_url,
            )
            info = wake_laptop(ep)
            self._set_boot_state(
                state=info.get("state", "waking"),
                stage=info.get("stage", "starting"),
                detail=info.get("detail", "Booting local models..."),
            )

            active_brain_url: str | None = None
            while not stop_event.is_set():
                st = poll_wake_status(ep)
                is_voice_ready = st.get("state") == "ready" or bool(st.get("voice_ready"))
                is_brain_ready = is_voice_ready or bool(st.get("brain_ready"))
                if is_brain_ready and active_brain_url is None:
                    active_brain_url = resolve_brain_url(ep)
                self._set_boot_state(
                    ready=is_voice_ready,
                    brain_ready=is_brain_ready,
                    voice_ready=is_voice_ready,
                    state=st.get("state", "waking"),
                    stage=st.get("stage", "starting"),
                    detail=st.get("detail", "Booting local models..."),
                    brain_url=active_brain_url or ep.brain_url,
                )
                if is_voice_ready:
                    break
                stop_event.wait(1.2)

            if stop_event.is_set():
                return

            if active_brain_url is None:
                active_brain_url = resolve_brain_url(ep)

            # 3. Silent wake: enable face tracking and welcoming head perk (zero uninvited speech).
            creature.set_tracking(True)
            creature.play_emotion("welcoming", force=True)

            # 4. Launch realtime audio + tool bridge in background.
            def _on_ws_state(connected: bool) -> None:
                self._set_boot_state(audio_connected=connected)
                self._send_heartbeat(
                    active_brain_url,
                    connected=connected,
                    mic_level=bridge.mic_level if bridge else 0.0,
                    mic_alive=bool(bridge and bridge.mic_alive),
                    mic_stalled=bool(bridge and bridge.mic_stalled),
                )

            initial_state = self._fetch_brain_state(active_brain_url) or {}
            initial_boost = float((initial_state.get("behavior") or {}).get("speaker_boost_db", 6.0))
            self._set_boot_state(speaker_boost_db=initial_boost)

            bridge = RealtimeAudioBridge(
                host=ep.host,
                port=ep.ws_port,
                reachy_mini=reachy_mini,
                creature=creature,
                on_connected=_on_ws_state,
                speaker_boost_db=initial_boost,
            )
            bridge_thread = threading.Thread(
                target=bridge.run_loop,
                args=(stop_event,),
                daemon=True,
                name="reachy-audio-bridge",
            )
            bridge_thread.start()

            # 5. Living-creature ambient loop + periodic heartbeat until Stop is pressed.
            last_hb = 0.0
            while not stop_event.is_set():
                now = time.monotonic()
                mic_lvl = bridge.mic_level if bridge else 0.0
                self._set_boot_state(
                    audio_connected=bool(bridge and bridge.connected),
                    mic_level=mic_lvl,
                    mic_alive=bool(bridge and bridge.mic_alive),
                    mic_stalled=bool(bridge and bridge.mic_stalled),
                    speaker_boost_db=bridge.speaker_boost_db if bridge else initial_boost,
                )
                if now - last_hb >= 5.0:
                    self._send_heartbeat(
                        active_brain_url,
                        connected=bool(bridge and bridge.connected),
                        mic_level=mic_lvl,
                        mic_alive=bool(bridge and bridge.mic_alive),
                        mic_stalled=bool(bridge and bridge.mic_stalled),
                    )
                    last_hb = now
                state = self._fetch_brain_state(active_brain_url)
                if state:
                    beh = state.get("behavior") or {}
                    if bridge is not None and isinstance(beh.get("speaker_boost_db"), (int, float)):
                        bridge.set_speaker_boost_db(float(beh["speaker_boost_db"]))
                    creature.tick_ambient(
                        ambient=state.get("ambient") or {},
                        behavior=beh,
                        muted=bool(state.get("muted", False)),
                        asleep=bool(state.get("asleep", False)),
                        speaking=bool(bridge and bridge.speaking),
                    )
                stop_event.wait(0.8)
        except KeyboardInterrupt:
            stop_event.set()
        finally:
            log.info("reachy_familiar stopping: cleaning up motion, session, and laptop GPU")
            creature.set_wobbling(False)
            creature.set_tracking(False)
            if ep is not None:
                final_brain_url = str(self.boot_status().get("brain_url") or ep.brain_url)
                self._send_heartbeat(final_brain_url, connected=False, detail="reachy_familiar stopped", mic_level=0.0)
                if os.environ.get("REACHY_FAMILIAR_KEEP_LAPTOP_AWAKE") != "1":
                    sleep_laptop(ep)
            if bridge_thread is not None:
                bridge_thread.join(timeout=3.0)
            web.stop()

    @staticmethod
    def _fetch_brain_state(brain_url: str) -> dict[str, Any] | None:
        try:
            r = requests.get(f"{brain_url.rstrip('/')}/api/state", timeout=1.5)
            if r.ok:
                return r.json()
        except Exception:
            pass
        return None

    @staticmethod
    def _send_heartbeat(
        brain_url: str,
        *,
        connected: bool,
        detail: str | None = None,
        mic_level: float = 0.0,
        mic_alive: bool = False,
        mic_stalled: bool = False,
    ) -> None:
        try:
            requests.post(
                f"{brain_url.rstrip('/')}/api/robot/heartbeat",
                json={
                    "backend_connected": connected,
                    "host": "reachy-mini.local",
                    "app": "reachy_familiar",
                    "mic_level": round(float(mic_level), 3),
                    "mic_alive": bool(mic_alive),
                    "mic_stalled": bool(mic_stalled),
                    "detail": detail or ("connected via reachy_familiar" if connected else "reconnecting"),
                },
                timeout=1.5,
            )
        except Exception:
            pass


if __name__ == "__main__":
    ReachyFamiliarApp().wrapped_run()
