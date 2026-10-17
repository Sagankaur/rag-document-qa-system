import hashlib

import numpy as np

from rag_pipeline import MinimalRAG


class FakeEmbeddingModel:
    """Small deterministic embedder for offline unit tests."""

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([byte / 255 for byte in digest[:8]])
        return np.asarray(vectors, dtype="float32")


def test_chunk_text_uses_overlap():
    rag = MinimalRAG(FakeEmbeddingModel(), chunk_size=4, chunk_overlap=1)

    assert rag.chunk_text("one two three four five six seven") == [
        "one two three four",
        "four five six seven",
        "seven",
    ]


def test_invalid_chunk_settings_are_rejected():
    try:
        MinimalRAG(FakeEmbeddingModel(), chunk_size=10, chunk_overlap=10)
    except ValueError as error:
        assert "smaller than chunk_size" in str(error)
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")


def test_build_and_retrieve_caps_top_k():
    rag = MinimalRAG(FakeEmbeddingModel(), chunk_size=3, chunk_overlap=0)

    assert rag.build_index("alpha beta gamma delta epsilon zeta") is True
    results = rag.retrieve("alpha", top_k=10)

    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)


def test_no_key_returns_retrieval_only_answer():
    rag = MinimalRAG(FakeEmbeddingModel())

    answer = rag.generate_answer("When does it open?", ["It opens at nine."])

    assert "Retrieval-only mode" in answer
    assert "It opens at nine." in answer
