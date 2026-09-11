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
