from __future__ import annotations

import math
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


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
