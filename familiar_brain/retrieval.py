from __future__ import annotations

import json
import sqlite3
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
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


@dataclass
class _CorpusCache:
    sources: list[str]
    titles: list[str]
    texts: list[str]
    metas: list[dict]
    weights: np.ndarray
    matrix: np.ndarray
    norms_sq: np.ndarray

    @property
    def count(self) -> int:
        return len(self.sources)


def _pack(v: Sequence[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


class Store:
    def __init__(self, path: Path | str, *, embedder: Embedder):
        self.embedder = embedder
        # One connection shared by request threads, the live indexer and the learner. A
        # Python sqlite3 connection holds one transaction at a time, so without the lock a
        # delete in one thread could commit or roll back another's half-done insert.
        # Embedding stays outside it: it is the slow part, and it touches no table.
        self._lock = threading.RLock()
        self._cache: dict[str, _CorpusCache] = {}
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
            for corpus in {c.corpus for c in chunks}:
                self._cache.pop(corpus, None)
        return len(chunks)

    def _get_corpus_cache_locked(self, corpus: str) -> _CorpusCache:
        cached = self._cache.get(corpus)
        if cached is not None:
            return cached
        rows = self.db.execute("""
            SELECT c.source, c.title, c.text, c.meta, c.weight, v.embedding
            FROM chunks c JOIN vec v ON v.chunk_id = c.id
            WHERE c.corpus = ?
            ORDER BY c.id""", (corpus,)).fetchall()
        if not rows:
            dim = self.embedder.dim
            cached = _CorpusCache(
                sources=[],
                titles=[],
                texts=[],
                metas=[],
                weights=np.empty((0,), dtype=np.float32),
                matrix=np.empty((0, dim), dtype=np.float32),
                norms_sq=np.empty((0,), dtype=np.float32),
            )
        else:
            sources = [r[0] for r in rows]
            titles = [r[1] for r in rows]
            texts = [r[2] for r in rows]
            metas = [json.loads(r[3]) for r in rows]
            weights = np.array([r[4] for r in rows], dtype=np.float32)
            dim = self.embedder.dim
            matrix = np.frombuffer(b"".join(r[5] for r in rows), dtype=np.float32).reshape(len(rows), dim).copy()
            norms_sq = np.einsum("ij,ij->i", matrix, matrix)
            cached = _CorpusCache(
                sources=sources,
                titles=titles,
                texts=texts,
                metas=metas,
                weights=weights,
                matrix=matrix,
                norms_sq=norms_sq,
            )
        self._cache[corpus] = cached
        return cached

    def _search_vec_locked(self, q_arr: np.ndarray, q_norm_sq: float, corpus: str, k: int) -> list[Hit]:
        if k <= 0:
            return []
        cache = self._get_corpus_cache_locked(corpus)
        if cache.count == 0:
            return []
        dots = cache.matrix @ q_arr
        dist_sq = np.maximum(np.float32(0.0), cache.norms_sq + np.float32(q_norm_sq) - np.float32(2.0) * dots)
        approx_scores = (np.float32(1.0) - np.sqrt(dist_sq) * np.float32(0.5)) * cache.weights
        cand_n = min(cache.count, max(k * 4, 64))
        cand_idx = np.argsort(-approx_scores, kind="stable")[:cand_n]
        exact_dists = np.linalg.norm(cache.matrix[cand_idx] - q_arr, axis=1)
        exact_scores = (1.0 - exact_dists * 0.5) * cache.weights[cand_idx]
        ranked = sorted(
            ((float(score), int(idx)) for score, idx in zip(exact_scores, cand_idx)),
            key=lambda pair: (-pair[0], pair[1]),
        )[:k]
        return [
            Hit(
                corpus=corpus,
                source=cache.sources[idx],
                title=cache.titles[idx],
                text=cache.texts[idx],
                meta=cache.metas[idx],
                score=score,
            )
            for score, idx in ranked
        ]

    def search(
        self,
        query: str,
        *,
        corpus: str,
        k: int = 3,
        query_vec: Sequence[float] | None = None,
    ) -> list[Hit]:
        """Nearest chunks within one corpus, re-ranked by weight."""
        if k <= 0:
            return []
        q = query_vec if query_vec is not None else self.embedder.embed([query])[0]
        q_arr = np.asarray(q, dtype=np.float32)
        q_norm_sq = float(np.dot(q_arr, q_arr))
        with self._lock:
            return self._search_vec_locked(q_arr, q_norm_sq, corpus, k)

    def search_corpora(
        self,
        query: str,
        corpora: Sequence[str],
        *,
        k: int = 3,
        query_vec: Sequence[float] | None = None,
    ) -> dict[str, list[Hit]]:
        """Embed `query` once and return top-k weighted hits for each requested corpus."""
        if not corpora:
            return {}
        if k <= 0:
            return {c: [] for c in corpora}
        q = query_vec if query_vec is not None else self.embedder.embed([query])[0]
        q_arr = np.asarray(q, dtype=np.float32)
        q_norm_sq = float(np.dot(q_arr, q_arr))
        with self._lock:
            return {c: self._search_vec_locked(q_arr, q_norm_sq, c, k) for c in corpora}

    def _knn(self, q: Sequence[float], k: int, corpus: str) -> list[tuple]:
        return self.db.execute("""
            SELECT c.corpus, c.source, c.title, c.text, c.meta, c.weight, v.distance
            FROM vec v JOIN chunks c ON c.id = v.chunk_id
            WHERE v.embedding MATCH ? AND k = ? AND v.corpus = ?
            ORDER BY v.distance""", (_pack(q), max(k * 4, 16), corpus)).fetchall()

    def delete_source(self, source: str) -> int:
        return self._delete("SELECT id FROM chunks WHERE source = ?", (source,))

    def delete_corpus(self, corpus: str) -> int:
        """Delete every chunk belonging to one corpus (used when switching campaign profiles)."""
        return self._delete("SELECT id FROM chunks WHERE corpus = ?", (corpus,))

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
            corpora = [r[0] for r in self.db.execute(f"SELECT DISTINCT corpus FROM chunks WHERE id IN ({qs})", ids)]
            self.db.execute(f"DELETE FROM vec WHERE chunk_id IN ({qs})", ids)
            self.db.execute(f"DELETE FROM chunks WHERE id IN ({qs})", ids)
            for corpus in corpora:
                self._cache.pop(corpus, None)
        return len(ids)

    def sources(self, corpus: str) -> dict[str, float]:
        """source -> max created_at, used by the notes indexer to skip unchanged files."""
        with self._lock:
            return {r[0]: r[1] for r in self.db.execute(
                "SELECT source, MAX(created_at) FROM chunks WHERE corpus = ? GROUP BY source", (corpus,))}

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            self.db.close()
