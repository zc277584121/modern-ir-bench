from __future__ import annotations

import sys
from types import ModuleType
from typing import ClassVar

from modern_ir_bench.rerankers import OpenRouterReranker


def test_openrouter_reranker_restores_input_order(monkeypatch) -> None:
    calls = []

    class Response:
        status_code = 200
        text = ""
        headers: ClassVar[dict[str, str]] = {}

        def json(self):
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }

    def post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return Response()

    requests = ModuleType("requests")
    requests.post = post
    requests.RequestException = OSError
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")

    reranker = OpenRouterReranker(model="cohere/rerank-test", request_concurrency=1)
    scores = reranker.score_batch(["query"], [["first", "second"]])

    assert scores == [[0.2, 0.9]]
    assert calls[0][0] == "https://openrouter.ai/api/v1/rerank"
    assert calls[0][2] == {
        "model": "cohere/rerank-test",
        "query": "query",
        "documents": ["first", "second"],
        "top_n": 2,
    }
    assert calls[0][1]["Authorization"] == "Bearer secret"
