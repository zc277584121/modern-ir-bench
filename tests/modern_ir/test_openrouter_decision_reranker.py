from __future__ import annotations

import sys
from types import ModuleType
from typing import ClassVar

import pytest

from modern_ir_bench.rerankers import (
    OpenRouterGroupedQuestionDecisionReranker,
    OpenRouterPointwiseDecisionReranker,
)


class _Response:
    status_code = 200
    text = ""
    headers: ClassVar[dict[str, str]] = {}

    def __init__(self, answers, *, cost: float = 0.001) -> None:
        self.answers = answers
        self.cost = cost

    def json(self):
        return {
            "answers": self.answers,
            "usage": {"input_tokens": 10, "output_tokens": len(self.answers), "cost": self.cost},
        }


def _requests(monkeypatch, post) -> None:
    requests = ModuleType("requests")
    requests.post = post
    requests.RequestException = OSError
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")


def test_pointwise_decision_reranker_scores_pairs_independently(monkeypatch) -> None:
    calls = []

    def post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        score = 0.9 if json["state"]["document"] == "relevant" else 0.1
        return _Response({"relevance": {"type": "noul", "noul": score}})

    _requests(monkeypatch, post)
    reranker = OpenRouterPointwiseDecisionReranker(model="perplexity/test", request_concurrency=1)

    assert reranker.score_batch(["query"], [["irrelevant", "relevant"]]) == [[0.1, 0.9]]
    assert len(calls) == 2
    assert calls[0][0] == "https://openrouter.ai/api/alpha/decisions"
    assert calls[0][1]["Authorization"] == "Bearer secret"
    assert calls[0][2]["state"] == {"query": "query", "document": "irrelevant"}
    assert reranker.metadata["usage"] == {
        "requests": 2,
        "input_tokens": 20,
        "output_tokens": 2,
        "cost_usd": pytest.approx(0.002),
    }


def test_pointwise_decision_reranker_rejects_invalid_probability(monkeypatch) -> None:
    _requests(monkeypatch, lambda *args, **kwargs: _Response({"relevance": {"noul": 1.2}}))
    reranker = OpenRouterPointwiseDecisionReranker(model="perplexity/test", request_concurrency=1)

    with pytest.raises(RuntimeError, match="invalid probability"):
        reranker.score_batch(["query"], [["document"]])


def test_grouped_question_reranker_isolates_documents_and_restores_order(monkeypatch, tmp_path) -> None:
    calls = []

    def post(url, *, headers, json, timeout):
        calls.append(json)
        answers = {
            name: {
                "type": "noul",
                "noul": 0.9 if question["instructions"]["candidate_document"] == "relevant" else 0.1,
            }
            for name, question in json["questions"].items()
        }
        return _Response(answers)

    _requests(monkeypatch, post)
    reranker = OpenRouterGroupedQuestionDecisionReranker(
        model="typesafe/test",
        group_size=2,
        request_cache_path=tmp_path / "groups.sqlite3",
        request_cache_namespace="test",
        request_concurrency=1,
    )

    assert reranker.score_batch(["query"], [["irrelevant one", "relevant", "irrelevant two"]]) == [[0.1, 0.9, 0.1]]
    assert len(calls) == 2
    assert sorted(len(call["questions"]) for call in calls) == [1, 2]
    assert calls[0]["state"] == {"query": "query"}
    assert calls[0]["questions"]["relevance_0"]["instructions"]["candidate_document"] == "irrelevant one"
    assert reranker.pairwise_cacheable is False
    reranker.close()


def test_grouped_question_reranker_resumes_from_exact_request_cache(monkeypatch, tmp_path) -> None:
    calls = []

    def post(url, *, headers, json, timeout):
        calls.append(json)
        return _Response({name: {"noul": index / 10} for index, name in enumerate(json["questions"], start=1)})

    _requests(monkeypatch, post)
    cache_path = tmp_path / "groups.sqlite3"
    first = OpenRouterGroupedQuestionDecisionReranker(
        model="typesafe/test",
        group_size=2,
        request_cache_path=cache_path,
        request_cache_namespace="test",
        request_concurrency=1,
    )
    assert first.score_batch(["query"], [["one", "two"]]) == [[0.1, 0.2]]
    first.close()

    second = OpenRouterGroupedQuestionDecisionReranker(
        model="typesafe/test",
        group_size=2,
        request_cache_path=cache_path,
        request_cache_namespace="test",
        request_concurrency=1,
    )
    assert second.score_batch(["query"], [["one", "two"]]) == [[0.1, 0.2]]
    assert len(calls) == 1
    assert second.metadata["request_cache"] == {"namespace": "test", "hits": 1, "misses": 0}
    second.close()
