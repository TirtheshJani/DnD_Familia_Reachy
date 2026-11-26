"""Health checks shared by scripts/run_s2s.py and the wake daemon.

Lives in the brain package so the wake daemon does not import from scripts/.
"""
from __future__ import annotations

from typing import Any

# How long a pool unit may sit in "draining" before s2s counts as wedged (S2S_STUCK_EXIT_S).
DEFAULT_STUCK_EXIT_S = 45.0


def pool_is_wedged(pool_json: dict[str, Any], max_drain_s: float) -> bool:
    """True when s2s's /v1/pool shows a unit that will not come back on its own.

    speech-to-speech 1.0.0 reports each unit as idle, active, draining (with
    draining_for_s) or stuck (its drain wait gave up and a handler may be dead, so
    the unit stays occupied forever and every new client gets session_limit_reached).
    """
    for unit in pool_json.get("units") or []:
        state = unit.get("state")
        if state == "stuck":
            return True
        if state == "draining" and float(unit.get("draining_for_s") or 0.0) >= max_drain_s:
            return True
    return False
