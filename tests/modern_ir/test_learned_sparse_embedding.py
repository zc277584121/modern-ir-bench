from __future__ import annotations

from typing import Any

import pytest
import torch

from modern_ir_bench.embeddings import (
    GteMultilingualSparseEmbedding,
    SentenceTransformersSparseEmbedding,
)


class _SparseEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def encode_document(self, texts: list[str], **_: Any) -> torch.Tensor:
        self.calls.append(("document", texts))
        return torch.tensor([[0.0, 2.0, 0.0], [1.5, 0.0, 3.0]]).to_sparse()

    def encode_query(self, texts: list[str], **_: Any) -> torch.Tensor:
        self.calls.append(("query", texts))
        return torch.tensor([[4.0, 0.0, 1.0]]).to_sparse()


def test_sentence_transformers_sparse_embedding_preserves_sparse_coordinates() -> None:
    embedding = SentenceTransformersSparseEmbedding(model="example/model", revision="revision")
    model = _SparseEncoder()
    embedding._model = model

    documents = embedding.encode_sparse_documents(["one", "two"])
    queries = embedding.encode_sparse_queries(["query"])

    assert documents == [{1: 2.0}, {0: 1.5, 2: 3.0}]
    assert queries == [{0: 4.0, 2: 1.0}]
    assert model.calls == [("document", ["one", "two"]), ("query", ["query"])]


def test_sentence_transformers_sparse_embedding_handles_empty_input() -> None:
    embedding = SentenceTransformersSparseEmbedding(model="example/model", revision="revision")

    assert embedding.encode_sparse_documents([]) == []
    assert embedding.encode_sparse_queries([]) == []
    assert embedding._model is None


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (SentenceTransformersSparseEmbedding, {"model": "", "revision": "revision"}),
        (SentenceTransformersSparseEmbedding, {"model": "model", "revision": "", "batch_size": 0}),
        (
            SentenceTransformersSparseEmbedding,
            {"model": "model", "revision": "revision", "pooling_chunk_size": 0},
        ),
        (GteMultilingualSparseEmbedding, {"revision": "", "batch_size": 1}),
        (GteMultilingualSparseEmbedding, {"revision": "revision", "max_length": 0}),
    ],
)
def test_learned_sparse_embedding_rejects_invalid_configuration(factory: Any, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        factory(**kwargs)
