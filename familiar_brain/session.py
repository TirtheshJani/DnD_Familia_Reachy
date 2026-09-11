from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path


class SessionLog:
    """Append-only JSONL log of every table utterance the brain hears."""

    def __init__(self, sessions_dir: Path | str):
        self.dir = Path(sessions_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.active_id: str | None = None
        self._lock = threading.Lock()

    def start(self) -> str:
        self.active_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path.touch()
        return self.active_id

    @property
    def path(self) -> Path:
        if self.active_id is None:
            raise RuntimeError("no active session")
        return self.dir / f"{self.active_id}.jsonl"

    def append(self, *, text: str, decision: str, spoke: bool, tag: str | None = None) -> None:
        rec = {"ts": time.time(), "text": text, "decision": decision, "spoke": spoke}
        if tag:
            rec["tag"] = tag
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def end(self) -> Path:
        p = self.path
        self.active_id = None
        return p
