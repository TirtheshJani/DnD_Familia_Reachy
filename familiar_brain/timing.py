from __future__ import annotations

import json
import logging
import time

log = logging.getLogger("familiar.timing")


class TurnTimer:
    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self._marks[f"{name}_ms"] = round((now - self._last) * 1000, 1)
        self._last = now

    def finish(self, *, decision: str, spoke: bool) -> dict:
        rec = dict(self._marks)
        rec["total_ms"] = round((time.perf_counter() - self._t0) * 1000, 1)
        rec["decision"] = decision
        rec["spoke"] = spoke
        log.info(json.dumps(rec))
        return rec
