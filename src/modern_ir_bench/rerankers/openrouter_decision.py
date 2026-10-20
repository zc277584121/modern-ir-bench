"""OpenRouter decision models used as relevance rerankers."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep
from typing import Any

from modern_ir_bench.rerankers.protocols import validate_batch

_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 524, 529}
_RELEVANCE_CRITERIA = {
    "true": "The document directly answers the query or provides material evidence needed to satisfy it.",
    "false": "The document is irrelevant, only superficially related, contradictory, or insufficient.",
}


class _OpenRouterDecisionClient:
    def __init__(self, *, model: str, request_concurrency: int, timeout_seconds: float, max_retries: int) -> None:
        if not model or request_concurrency < 1 or timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("Model, concurrency, timeout, and retries must be valid")
        self.model = model
        self.request_concurrency = request_concurrency
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.request_count = 0

    def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        import requests

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter decision models")
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/alpha/decisions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                sleep(min(60, 2**attempt))
                continue
            if response.status_code == 200:
                body = response.json()
                if not isinstance(body, Mapping):
                    raise RuntimeError("OpenRouter decision model returned an invalid response")
                return body
            if response.status_code not in _RETRYABLE_STATUS_CODES or attempt == self.max_retries:
                raise RuntimeError(
                    f"OpenRouter decision request failed with HTTP {response.status_code}: {response.text[:500]}"
                )
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(60, 2**attempt)
            except ValueError:
                delay = min(60, 2**attempt)
            sleep(min(60, max(0, delay)))
        raise AssertionError("unreachable")

    @staticmethod
    def _read_probability(body: Mapping[str, Any], question: str) -> float:
        try:
            score = float(body["answers"][question]["noul"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("OpenRouter decision model returned an invalid relevance answer") from error
        if not 0 <= score <= 1:
            raise RuntimeError(f"OpenRouter decision model returned an invalid probability: {score}")
        return score

    def _record_usage(self, body: Mapping[str, Any]) -> None:
        usage = body.get("usage", {})
        if not isinstance(usage, Mapping):
            usage = {}
        self.input_tokens += int(usage.get("input_tokens", 0))
        self.output_tokens += int(usage.get("output_tokens", 0))
        self.cost_usd += float(usage.get("cost", 0.0))
        self.request_count += 1

    @property
    def _usage_metadata(self) -> Mapping[str, Any]:
        return {
            "requests": self.request_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


class OpenRouterPointwiseDecisionReranker(_OpenRouterDecisionClient):
    """Score every query-document pair with an independent decision request."""

    pairwise_cacheable = True

    def __init__(
        self,
        *,
        model: str,
        request_concurrency: int = 48,
        timeout_seconds: float = 300,
        max_retries: int = 6,
    ) -> None:
        super().__init__(
            model=model,
            request_concurrency=request_concurrency,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        self.id = f"openrouter-decision:{model}:pointwise-noul-relevance-v1"

    def score_batch(self, queries: Sequence[str], documents: Sequence[Sequence[str]]) -> list[list[float]]:
        validate_batch(queries, documents)
        jobs = [
            (query_index, document_index, query, document)
            for query_index, (query, rows) in enumerate(zip(queries, documents, strict=True))
            for document_index, document in enumerate(rows)
        ]
        scores = [[0.0] * len(rows) for rows in documents]
        with ThreadPoolExecutor(max_workers=self.request_concurrency) as executor:
            futures = [executor.submit(self._score_one, query, document) for _, _, query, document in jobs]
            for (query_index, document_index, _, _), future in zip(jobs, futures, strict=True):
                score, body = future.result()
                scores[query_index][document_index] = score
                self._record_usage(body)
        return scores

    def _score_one(self, query: str, document: str) -> tuple[float, Mapping[str, Any]]:
        body = self._post(
            {
                "model": self.model,
                "state": {"query": query, "document": document},
                "questions": {
                    "relevance": {
                        "type": "noul",
                        "instructions": (
                            "Is the document relevant to the query and useful for satisfying the information need?"
                        ),
                        "criteria": _RELEVANCE_CRITERIA,
                    }
                },
            }
        )
        return self._read_probability(body, "relevance"), body

    def close(self) -> None:
        """Requests do not hold process-level resources."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "openrouter-decisions",
            "model": self.model,
            "primitive": "noul",
            "protocol": "pointwise",
            "scoring": "relevance-probability",
            "request_concurrency": self.request_concurrency,
            "usage": self._usage_metadata,
        }


class OpenRouterGroupedQuestionDecisionReranker(_OpenRouterDecisionClient):
    """Score isolated document questions together in bounded API requests."""

    pairwise_cacheable = False

    def __init__(
        self,
        *,
        model: str,
        group_size: int,
        request_cache_path: Path,
        request_cache_namespace: str,
        request_concurrency: int = 24,
        timeout_seconds: float = 300,
        max_retries: int = 6,
    ) -> None:
        if group_size < 1 or not request_cache_namespace:
            raise ValueError("Group size and request cache namespace must be configured")
        super().__init__(
            model=model,
            request_concurrency=request_concurrency,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        self.group_size = group_size
        self.request_cache_namespace = request_cache_namespace
        self.id = f"openrouter-decision:{model}:grouped-questions-{group_size}-noul-relevance-v1"
        request_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = sqlite3.connect(request_cache_path)
        self._cache.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_group_scores (
                namespace TEXT NOT NULL,
                request_key TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                PRIMARY KEY (namespace, request_key)
            )
            """
        )
        self._cache.commit()
        self.request_cache_hits = 0
        self.request_cache_misses = 0

    def score_batch(self, queries: Sequence[str], documents: Sequence[Sequence[str]]) -> list[list[float]]:
        validate_batch(queries, documents)
        jobs = [
            (query_index, query, offset, list(rows[offset : offset + self.group_size]))
            for query_index, (query, rows) in enumerate(zip(queries, documents, strict=True))
            for offset in range(0, len(rows), self.group_size)
        ]
        scores = [[0.0] * len(rows) for rows in documents]
        pending = []
        for query_index, query, offset, group in jobs:
            request_key = self._request_key(query, group)
            cached = self._read_cache(request_key)
            if cached is None:
                self.request_cache_misses += 1
                pending.append((query_index, query, offset, group, request_key))
            else:
                self.request_cache_hits += 1
                scores[query_index][offset : offset + len(group)] = cached

        with ThreadPoolExecutor(max_workers=self.request_concurrency) as executor:
            futures = [executor.submit(self._score_group, query, group) for _, query, _, group, _ in pending]
            for (query_index, _, offset, group, request_key), future in zip(pending, futures, strict=True):
                group_scores, body = future.result()
                scores[query_index][offset : offset + len(group)] = group_scores
                self._write_cache(request_key, group_scores)
                self._record_usage(body)
        return scores

    def _score_group(self, query: str, documents: Sequence[str]) -> tuple[list[float], Mapping[str, Any]]:
        questions = {
            f"relevance_{offset}": {
                "type": "noul",
                "instructions": {
                    "candidate_document": document,
                    "question": (
                        "Is `candidate_document` relevant to `query` and useful for satisfying the information need? "
                        "Judge only this candidate document."
                    ),
                },
                "criteria": _RELEVANCE_CRITERIA,
            }
            for offset, document in enumerate(documents)
        }
        body = self._post({"model": self.model, "state": {"query": query}, "questions": questions})
        answers = body.get("answers")
        if not isinstance(answers, Mapping) or set(answers) != set(questions):
            raise RuntimeError("OpenRouter decision model returned incomplete grouped answers")
        return [self._read_probability(body, name) for name in questions], body

    def _request_key(self, query: str, documents: Sequence[str]) -> str:
        payload = json.dumps(
            {"reranker_id": self.id, "query": query, "documents": list(documents)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _read_cache(self, request_key: str) -> list[float] | None:
        row = self._cache.execute(
            "SELECT scores_json FROM decision_group_scores WHERE namespace = ? AND request_key = ?",
            (self.request_cache_namespace, request_key),
        ).fetchone()
        return [float(value) for value in json.loads(row[0])] if row is not None else None

    def _write_cache(self, request_key: str, scores: Sequence[float]) -> None:
        self._cache.execute(
            "INSERT OR REPLACE INTO decision_group_scores (namespace, request_key, scores_json) VALUES (?, ?, ?)",
            (self.request_cache_namespace, request_key, json.dumps(list(scores), separators=(",", ":"))),
        )
        self._cache.commit()

    def close(self) -> None:
        self._cache.close()

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "openrouter-decisions",
            "model": self.model,
            "primitive": "noul",
            "protocol": "grouped-questions",
            "group_size": self.group_size,
            "scoring": "relevance-probability",
            "request_concurrency": self.request_concurrency,
            "request_cache": {
                "namespace": self.request_cache_namespace,
                "hits": self.request_cache_hits,
                "misses": self.request_cache_misses,
            },
            "usage": self._usage_metadata,
        }
