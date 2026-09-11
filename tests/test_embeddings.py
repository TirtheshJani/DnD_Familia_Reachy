import pytest

from familiar_brain.embeddings import Embedder, cosine
from tests.fakes.embedder import FakeEmbedder


def test_fake_embedder_satisfies_protocol_and_similarity_orders():
    e: Embedder = FakeEmbedder()
    a, b, c = e.embed(["grapple the target", "grapple a creature", "cast fireball"])
    assert cosine(a, b) > cosine(a, c)


@pytest.mark.slow
def test_bge_embedder_loads_on_cpu_and_has_384_dims():
    pytest.importorskip("sentence_transformers")
    from familiar_brain.embeddings import BgeEmbedder
    e = BgeEmbedder()
    v = e.embed(["hello"])[0]
    assert len(v) == 384 and e.dim == 384


@pytest.mark.slow
def test_open_store_returns_an_already_loaded_embedder(tmp_path):
    """open_store's contract is a store that is ready to search, not one that will
    spend twelve seconds loading the first time somebody speaks."""
    pytest.importorskip("sentence_transformers")
    from types import SimpleNamespace
    from familiar_brain.indexing import open_store
    cfg = SimpleNamespace(knowledge=SimpleNamespace(db_path=str(tmp_path / "k.db")))
    store = open_store(cfg)
    assert store.embedder._model is not None
