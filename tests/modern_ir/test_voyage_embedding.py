from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from modern_ir_bench.embeddings import VoyageEmbedding


class FakeVoyageClient:
    def __init__(self) -> None:
        self.requests = []

    def tokenize(self, texts, *, model):
        return [SimpleNamespace(ids=list(range(len(text.split())))) for text in texts]

    def embed(self, texts, *, model, input_type, output_dimension):
        self.requests.append((list(texts), model, input_type, output_dimension))
        return SimpleNamespace(embeddings=[[float(len(text)), float(index)] for index, text in enumerate(texts)])


def test_voyage_embedding_batches_by_tokens_and_reuses_disk_cache(tmp_path) -> None:
    embedding = VoyageEmbedding(
        model="voyage-test",
        dimension=2,
        batch_size=3,
        max_batch_tokens=3,
        cache_path=tmp_path / "voyage.sqlite3",
    )
    client = FakeVoyageClient()
    embedding._client = client

    first = embedding.encode_documents(["one two", "three four", "five"])
    second = embedding.encode_documents(["one two", "three four", "five"])

    assert first.shape == (3, 2)
    assert np.array_equal(first, second)
    assert [request[0] for request in client.requests] == [
        ["one two"],
        ["three four", "five"],
    ]
    assert {request[2] for request in client.requests} == {"document"}
    assert {request[3] for request in client.requests} == {2}


def test_voyage_embedding_returns_shaped_empty_array() -> None:
    embedding = VoyageEmbedding(model="voyage-test", dimension=2)

    result = embedding.encode_queries([])

    assert result.shape == (0, 2)
