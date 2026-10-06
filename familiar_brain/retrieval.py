from __future__ import annotations

import json
import sqlite3
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path

import sqlite_vec

from familiar_brain.embeddings import Embedder


@dataclass
class Chunk:
    corpus: str
    source: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class Hit:
    corpus: str
    source: str
    title: str
    text: str
    meta: dict
    score: float


def _pack(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


class Store:
    def __init__(self, path: Path | str, *, embedder: Embedder):
        self.embedder = embedder
        # One connection shared by request threads, the live indexer and the learner. A
        # Python sqlite3 connection holds one transaction at a time, so without the lock a
        # delete in one thread could commit or roll back another's half-done insert.
        # Embedding stays outside it: it is the slow part, and it touches no table.
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.executescript(f"""
        CREATE TABLE IF NOT EXISTS chunks(
          id INTEGER PRIMARY KEY, corpus TEXT NOT NULL, source TEXT NOT NULL, title TEXT,
          text TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{{}}', weight REAL NOT NULL DEFAULT 1.0,
          created_at REAL DEFAULT (unixepoch('subsec')));
        CREATE INDEX IF NOT EXISTS chunks_corpus ON chunks(corpus);
        CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
        CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0(chunk_id INTEGER PRIMARY KEY, corpus TEXT, embedding float[{embedder.dim}]);
        """)

    def add_chunks(self, chunks: list[Chunk], *, replace: tuple[str, ...] = ()) -> int:
        """Embed and insert `chunks`. Any `replace` sources are deleted in the same
        transaction, after the embedding has succeeded, so a failure leaves the old chunks
        searchable: the summariser deleted a session's live chunks first, and an embedder
        failure after that left the session with nothing in the store at all."""
        if not chunks and not replace:
            return 0
        vecs = self.embedder.embed([f"{c.title}\n{c.text}" for c in chunks]) if chunks else []
        with self._lock, self.db:
            for source in replace:
                self._delete_in_tx("SELECT id FROM chunks WHERE source = ?", (source,))
            for c, v in zip(chunks, vecs):
                cur = self.db.execute(
                    "INSERT INTO chunks(corpus, source, title, text, meta, weight) VALUES (?,?,?,?,?,?)",
                    (c.corpus, c.source, c.title, c.text, json.dumps(c.meta), c.weight))
                self.db.execute("INSERT INTO vec(chunk_id, corpus, embedding) VALUES (?,?,?)",
                                (cur.lastrowid, c.corpus, _pack(v)))
        return len(chunks)

    def search(self, query: str, *, corpus: str, k: int = 3) -> list[Hit]:
        """Nearest chunks within one corpus, re-ranked by weight.

        `corpus` is a vec0 metadata column so the KNN search itself is scoped. Filtering
        a fixed candidate set afterwards instead loses small corpora completely: with
        2000 rules chunks indexed beside a handful of notes, every candidate came back
        a rules chunk and `search(corpus="notes")` returned nothing. Over-fetch beyond k
        is still needed, because weight can lift a slightly more distant chunk.
        """
        q = self.embedder.embed([query])[0]
        with self._lock:
            rows = self._knn(q, k, corpus)
        hits = [Hit(r[0], r[1], r[2], r[3], json.loads(r[4]), score=(1.0 - r[6] / 2.0) * r[5]) for r in rows]
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    def _knn(self, q: list[float], k: int, corpus: str) -> list[tuple]:
        return self.db.execute("""
            SELECT c.corpus, c.source, c.title, c.text, c.meta, c.weight, v.distance
            FROM vec v JOIN chunks c ON c.id = v.chunk_id
            WHERE v.embedding MATCH ? AND k = ? AND v.corpus = ?
            ORDER BY v.distance""", (_pack(q), max(k * 4, 16), corpus)).fetchall()

    def delete_source(self, source: str) -> int:
        return self._delete("SELECT id FROM chunks WHERE source = ?", (source,))

    def delete_pieces(self, source: str, first: int) -> int:
        """Drop the transcript pieces of `source` numbered `first` and up, leaving the
        pieces before them and any chunk with no piece number (a deliberate note).

        The live indexer's seam: when a line is added or revised only the tail of the
        transcript changes, and re-embedding the whole of a four hour session every batch
        would grow with the session for no gain (transcript.transcript_pieces).
        """
        return self._delete("SELECT id FROM chunks WHERE source = ? AND json_extract(meta, '$.piece') >= ?",
                            (source, first))

    def _delete(self, select: str, args: tuple) -> int:
        with self._lock, self.db:
            return self._delete_in_tx(select, args)

    def _delete_in_tx(self, select: str, args: tuple) -> int:
        ids = [r[0] for r in self.db.execute(select, args)]
        if ids:
            qs = ",".join("?" * len(ids))
            self.db.execute(f"DELETE FROM vec WHERE chunk_id IN ({qs})", ids)
            self.db.execute(f"DELETE FROM chunks WHERE id IN ({qs})", ids)
        return len(ids)

    def sources(self, corpus: str) -> dict[str, float]:
        """source -> max created_at, used by the notes indexer to skip unchanged files."""
        with self._lock:
            return {r[0]: r[1] for r in self.db.execute(
                "SELECT source, MAX(created_at) FROM chunks WHERE corpus = ? GROUP BY source", (corpus,))}
