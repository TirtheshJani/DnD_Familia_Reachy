from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class DeterministicEmbedder:
    """Fast, dependency-free CPU hash embedder producing L2-normalized 384-d vectors."""

    dim = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self.dim
            words = re.findall(r"[a-z0-9]+", t.lower())
            if not words:
                h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
                v[h % self.dim] = 1.0
            else:
                for w in words:
                    h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                    v[h % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class BgeEmbedder:
    """CPU-only sentence-transformers embedder. Model pinned; loaded lazily on first call.

    Lazy because importing sentence_transformers pulls torch, which costs seconds. The
    tests never touch it: they use tests/fakes/embedder.FakeEmbedder instead.
    """
    MODEL = "BAAI/bge-small-en-v1.5"
    dim = 384

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL, device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._load().encode(texts, normalize_embeddings=True, batch_size=32,
                                   show_progress_bar=False)
        return [v.tolist() for v in vecs]


def make_embedder(name: str = "bge") -> Embedder:
    kind = (name or "bge").strip().lower()
    if kind == "bge":
        return BgeEmbedder()
    if kind == "deterministic":
        return DeterministicEmbedder()
    raise ValueError(f"unknown embedder {name!r}; expected 'bge' or 'deterministic'")
