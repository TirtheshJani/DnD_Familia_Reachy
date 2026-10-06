"""Keeps the "sessions" corpus up to date with the session in progress, by embedding only.

So "Rocky, what did the innkeeper say earlier?" can be answered mid-session. No LLM call
runs here: live on 2026-09-16 the GPU was contended enough that classify calls timed out
at 1.5 s and a reply took 60 s. The BGE embedder runs on the CPU (21 ms warm, P11).

It reads the session log file, not the turns, so it indexes exactly what the log holds:
lines dropped by mute-and-stop-logging (app._log) never reach the file and so never reach
the store. Chunks go under the raw log's own name, the source summarise_session,
apply_retention and purge_session already delete, so the end-of-session summary replaces
them like for like and a purge forgets them.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

from familiar_brain.retrieval import Store
from familiar_brain.transcript import (
    DELIBERATE,
    fold_row,
    note_chunks,
    piece_chunks,
    transcript_pieces,
)

log = logging.getLogger("familiar.live_index")

# How often the worker looks at the log. Reading a few new lines costs nothing; this only
# bounds how soon a deliberate note becomes searchable.
TICK_S = 5.0
# Index once the oldest unindexed line is this old. app.REVISION_WINDOW_S is 30 s: most
# revisions of a line have landed by then, so the tail is embedded about once per settled
# utterance rather than once per re-send (2 to 6 per line on 2026-09-16). Asking about
# something said under half a minute ago is not a question for memory anyway.
BATCH_AGE_S = 30.0
# Or sooner, once this many new rows have piled up, so a burst of table talk is not held
# back. Each batch re-embeds only the changed tail piece, so a batch costs a piece or two
# however long the session has run.
BATCH_LINES = 8


def _spawn(target: Callable[[], None]) -> threading.Thread:
    t = threading.Thread(target=target, daemon=True, name="live-index")
    t.start()
    return t


class LiveIndexer:
    def __init__(self, store: Store, *, clock: Callable[[], float] = time.time,
                 spawn: Callable[[Callable[[], None]], object] | None = None):
        self.store = store
        self.clock = clock
        # Looked up at call time when not given, so a test can swap the module's _spawn.
        self._spawn = spawn
        self._lock = threading.Lock()
        self._stop: threading.Event | None = None
        self.path: Path | None = None

    def begin(self, path: Path) -> None:
        """Start indexing one session's log, and the worker that ticks it."""
        self.end()
        with self._lock:
            self.path = Path(path)
            self.sid = self.path.stem
            self._offset = 0
            self._lines: list[dict] = []
            self._notes: list[dict] = []        # deliberate notes not yet indexed
            self._indexed: list[str] = []       # the pieces in the store, in order
            self._pending = 0
            self._pending_since: float | None = None
            stop = self._stop = threading.Event()
        (self._spawn or _spawn)(lambda: self._run(stop))

    def _run(self, stop: threading.Event) -> None:
        while not stop.wait(TICK_S):
            self.tick()

    def end(self) -> None:
        """Index whatever is left and stop, before the log is handed to the summariser.

        Under the lock, so once this returns no tick can write this session's chunks again:
        a write landing after the summariser's delete_source would double them.
        """
        with self._lock:
            if self._stop is not None:
                self._stop.set()
                self._stop = None
            if self.path is None:
                return
            self._read()
            self._flush()
            self.path = None

    def tick(self, now: float | None = None) -> int:
        """Read new rows and index them if a batch is due. Returns chunks added."""
        with self._lock:
            if self.path is None:
                return 0
            now = self.clock() if now is None else now
            if self._read() and self._pending_since is None:
                self._pending_since = now
            due = (self._notes or self._pending >= BATCH_LINES
                   or (self._pending and now - self._pending_since >= BATCH_AGE_S))
            return self._flush() if due else 0

    def _read(self) -> int:
        """Fold the log's new whole lines in. A half-written last line is left for later."""
        try:
            with self.path.open("rb") as f:
                f.seek(self._offset)
                data = f.read()
        except FileNotFoundError:
            return 0
        end = data.rfind(b"\n") + 1
        self._offset += end
        n = 0
        for raw in data[:end].splitlines():
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not row.get("text"):
                continue
            if row.get("tag") == DELIBERATE:
                self._notes.append(row)
            fold_row(self._lines, row)
            n += 1
        self._pending += n
        return n

    def _flush(self) -> int:
        """Index pending notes, then the transcript pieces that changed. A failure leaves
        what failed pending, so the next tick tries again; a turn never waits on any of it."""
        source = self.path.name
        added = 0
        try:
            if self._notes:
                added += self.store.add_chunks(note_chunks(self._notes, sid=self.sid, source=source))
                self._notes = []
            pieces = transcript_pieces(self._lines)
            first = next((i for i, (old, new) in enumerate(zip(self._indexed, pieces)) if old != new),
                         min(len(self._indexed), len(pieces)))
            if first < len(self._indexed):
                self.store.delete_pieces(source, first)
                self._indexed = self._indexed[:first]
            if first < len(pieces):
                added += self.store.add_chunks(piece_chunks(pieces, sid=self.sid, source=source, first=first))
                self._indexed = pieces
        except Exception as e:
            log.warning("live indexing of %s failed, will retry: %s", source, e)
            return added
        self._pending, self._pending_since = 0, None
        if added:
            log.info("live indexed %d chunk(s) of %s", added, source)
        return added
