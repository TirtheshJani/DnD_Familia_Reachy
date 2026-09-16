"""Read-only client for the Reachy Mini daemon on the robot.

The conversation app owns the head, the antennas and the camera. The brain never
commands the daemon; it only reads the microphone array's direction of arrival so a
glance can be aimed at whoever just spoke. Every failure degrades to None inside the
gate's latency budget, because a glance is never worth a late reply.
"""
from __future__ import annotations

import logging
import math
from typing import Callable

import requests

log = logging.getLogger("familiar.robot")

# Below pi/2 minus this is left, above pi/2 plus this is right; the middle is front.
DEAD_ZONE_RAD = 0.35


def direction_from_angle(angle: float, dead_zone: float = DEAD_ZONE_RAD) -> str:
    """Map a DoA angle (0 left, pi/2 front, pi right) to a move_head direction."""
    if angle < math.pi / 2 - dead_zone:
        return "left"
    if angle > math.pi / 2 + dead_zone:
        return "right"
    return "front"


class RobotDaemon:
    def __init__(self, base_url: str, *, timeout_s: float = 0.15,
                 get: Callable[..., requests.Response] = requests.get):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.get = get

    def doa(self) -> tuple[float, bool] | None:
        """(angle_rad, speech_detected) from GET /api/state/doa, or None."""
        if not self.base_url:
            return None
        try:
            r = self.get(f"{self.base_url}/api/state/doa", timeout=self.timeout_s)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, dict) or "angle" not in payload:
                return None
            # Inside the try: an angle that is not a number is a failure like any other.
            angle = float(payload["angle"])
            if not math.isfinite(angle):  # NaN maps to "front" and would spend a glance
                return None
            return angle, bool(payload.get("speech_detected", False))
        except Exception as e:
            log.debug("doa unavailable: %s", type(e).__name__)
            return None
