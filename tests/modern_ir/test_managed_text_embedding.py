from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import numpy as np

from modern_ir_bench.embeddings import DashScopeTextEmbedding, JinaTextEmbedding


class Response:
    status_code = 200
    text = ""
    headers: ClassVar[dict[str, str]] = {}

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def json(self) -> dict[str, object]:
        return self.payload


def _requests(monkeypatch, post):
    requests = ModuleType("requests")
    requests.post = post
    requests.RequestException = OSError
    monkeypatch.setitem(sys.modules, "requests", requests)


def test_jina_uses_retrieval_tasks_and_persistent_cache(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return Response(
            {
                "data": [
                    {"index": index, "embedding": [float(index), 1.0]}
                    for index in range(len(json["input"]))
                ]
            }
        )

    _requests(monkeypatch, post)
    monkeypatch.setenv("JINA_API_KEY", "secret")
    embedding = JinaTextEmbedding(
        model="jina-test",
        dimension=2,
        batch_size=2,
        cache_path=tmp_path / "jina.sqlite3",
    )

    documents = embedding.encode_documents(["one", "two", "one"])
    queries = embedding.encode_queries(["one"])
    cached = embedding.encode_documents(["one", "two"])

    assert documents.shape == (3, 2)
    assert queries.shape == (1, 2)
    assert np.array_equal(cached, documents[:2])
    assert [call[2]["task"] for call in calls] == ["retrieval.passage", "retrieval.query"]
    assert calls[0][2]["dimensions"] == 2
    assert calls[0][1]["Authorization"] == "Bearer secret"


def test_dashscope_uses_native_query_and_document_modes(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return Response(
            {
                "output": {
                    "embeddings": [
                        {"text_index": index, "embedding": [float(index), 1.0]}
                        for index in range(len(json["input"]["texts"]))
                    ]
                }
            }
        )

    _requests(monkeypatch, post)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret")
    embedding = DashScopeTextEmbedding(
        model="dashscope-test",
        dimension=2,
        batch_size=2,
        cache_path=tmp_path / "dashscope.sqlite3",
        query_instruction="Retrieve relevant documents.",
    )

    assert embedding.encode_documents(["one", "two"]).shape == (2, 2)
    assert embedding.encode_queries(["one"]).shape == (1, 2)
    document_parameters = calls[0][2]["parameters"]
    query_parameters = calls[1][2]["parameters"]

    assert document_parameters == {"dimension": 2, "output_type": "dense", "text_type": "document"}
    assert query_parameters == {
        "dimension": 2,
        "output_type": "dense",
        "text_type": "query",
        "instruct": "Retrieve relevant documents.",
    }
