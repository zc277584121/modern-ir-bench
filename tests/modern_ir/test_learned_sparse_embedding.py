from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from modern_ir_bench.embeddings import (
    GteMultilingualSparseEmbedding,
    OnnxSpladeEmbedding,
    SentenceTransformersSparseEmbedding,
)


class _SparseEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.options: list[dict[str, Any]] = []

    def encode_document(self, texts: list[str], **options: Any) -> torch.Tensor:
        self.calls.append(("document", texts))
        self.options.append(options)
        return torch.tensor([[0.0, 2.0, 0.0], [1.5, 0.0, 3.0]]).to_sparse()

    def encode_query(self, texts: list[str], **options: Any) -> torch.Tensor:
        self.calls.append(("query", texts))
        self.options.append(options)
        return torch.tensor([[4.0, 0.0, 1.0]]).to_sparse()


class _OnnxTokenizer:
    def __call__(self, text: str, **options: Any) -> dict[str, np.ndarray]:
        assert text == "query"
        assert options == {"truncation": True, "max_length": 512, "return_tensors": "np"}
        return {
            "input_ids": np.asarray([[101, 23032, 102]]),
            "attention_mask": np.asarray([[1, 1, 1]]),
        }

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        assert token_ids == [23032, 100]
        return ["care", "##giver"]


class _OnnxSession:
    def run(self, outputs: Any, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        assert outputs is None
        assert inputs["input_ids"].dtype == np.int64
        return np.asarray([23032, 100, 200]), np.asarray([2.5, 0.0, -1.0])


def test_sentence_transformers_sparse_embedding_preserves_sparse_coordinates() -> None:
    embedding = SentenceTransformersSparseEmbedding(model="example/model", revision="revision")
    model = _SparseEncoder()
    embedding._model = model

    documents = embedding.encode_sparse_documents(["one", "two"])
    queries = embedding.encode_sparse_queries(["query"])

    assert documents == [{1: 2.0}, {0: 1.5, 2: 3.0}]
    assert queries == [{0: 4.0, 2: 1.0}]
    assert model.calls == [("document", ["one", "two"]), ("query", ["query"])]


def test_sentence_transformers_sparse_embedding_uses_separate_active_dimension_limits() -> None:
    embedding = SentenceTransformersSparseEmbedding(
        model="example/model",
        revision="revision",
        document_max_active_dimensions=192,
        query_max_active_dimensions=50,
    )
    model = _SparseEncoder()
    embedding._model = model

    embedding.encode_sparse_documents(["one", "two"])
    embedding.encode_sparse_queries(["query"])

    assert model.options[0]["max_active_dims"] == 192
    assert model.options[1]["max_active_dims"] == 50


def test_sentence_transformers_sparse_embedding_handles_empty_input() -> None:
    embedding = SentenceTransformersSparseEmbedding(model="example/model", revision="revision")

    assert embedding.encode_sparse_documents([]) == []
    assert embedding.encode_sparse_queries([]) == []
    assert embedding._model is None


def test_onnx_splade_embedding_preserves_positive_sparse_weights() -> None:
    embedding = OnnxSpladeEmbedding(model="example/model", revision="revision")
    embedding._session = _OnnxSession()
    embedding._tokenizer = _OnnxTokenizer()

    assert embedding.encode_sparse_queries(["query"]) == [{23032: 2.5}]
    assert embedding.tokens_for_ids([23032, 100]) == ["care", "##giver"]

    embedding.release()
    assert embedding._session is None
    assert embedding._tokenizer is None


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (SentenceTransformersSparseEmbedding, {"model": "", "revision": "revision"}),
        (SentenceTransformersSparseEmbedding, {"model": "model", "revision": "", "batch_size": 0}),
        (
            SentenceTransformersSparseEmbedding,
            {"model": "model", "revision": "revision", "pooling_chunk_size": 0},
        ),
        (
            SentenceTransformersSparseEmbedding,
            {"model": "model", "revision": "revision", "document_max_active_dimensions": 0},
        ),
        (GteMultilingualSparseEmbedding, {"revision": "", "batch_size": 1}),
        (GteMultilingualSparseEmbedding, {"revision": "revision", "max_length": 0}),
        (OnnxSpladeEmbedding, {"model": "", "revision": "revision"}),
        (OnnxSpladeEmbedding, {"model": "model", "revision": "revision", "max_length": 0}),
    ],
)
def test_learned_sparse_embedding_rejects_invalid_configuration(factory: Any, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        factory(**kwargs)
