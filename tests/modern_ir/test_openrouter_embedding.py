from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from modern_ir_bench.embeddings import OpenRouterEmbedding


def test_openrouter_uses_input_types_and_persistent_cache(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class Response:
        status_code = 200
        text = ""

        def __init__(self, count: int) -> None:
            self.count = count
            self.headers = {}

        def json(self):
            return {
                "data": [
                    {"index": index, "embedding": [float(index), 1.0]}
                    for index in range(self.count)
                ]
            }

    def post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return Response(len(json["input"]))

    requests = ModuleType("requests")
    requests.post = post
    requests.RequestException = OSError
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")

    embedding = OpenRouterEmbedding(
        model="provider/model",
        dimension=2,
        batch_size=2,
        cache_path=tmp_path / "embeddings.sqlite3",
        document_input_type="search_document",
        query_input_type="search_query",
    )
    documents = embedding.encode_documents(["one", "two", "one"])
    queries = embedding.encode_queries(["one"])
    cached_documents = embedding.encode_documents(["one", "two"])

    assert documents.shape == (3, 2)
    assert queries.shape == (1, 2)
    assert np.array_equal(cached_documents, documents[:2])
    assert [call[2]["input_type"] for call in calls] == ["search_document", "search_query"]
    assert calls[0][2]["dimensions"] == 2
    assert calls[0][1]["Authorization"] == "Bearer secret"
