"""OpenRouter-hosted rerankers with bounded concurrent requests."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from typing import Any

from modern_ir_bench.rerankers.protocols import validate_batch


class OpenRouterReranker:
    def __init__(
        self,
        *,
        model: str,
        request_concurrency: int = 8,
        timeout_seconds: float = 300,
        max_retries: int = 6,
    ) -> None:
        if not model or request_concurrency < 1 or timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("Model, concurrency, timeout, and retries must be valid")
        self.model = model
        self.id = f"openrouter:{model}"
        self.request_concurrency = request_concurrency
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def score_batch(
        self,
        queries: Sequence[str],
        documents: Sequence[Sequence[str]],
    ) -> list[list[float]]:
        validate_batch(queries, documents)
        with ThreadPoolExecutor(max_workers=self.request_concurrency) as executor:
            requests = [executor.submit(self._score_one, query, list(rows)) for query, rows in zip(queries, documents)]
            return [request.result() for request in requests]

    def _score_one(self, query: str, documents: list[str]) -> list[float]:
        import requests

        if not documents:
            return []
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter reranking")
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/rerank",
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
                scores: list[float | None] = [None] * len(documents)
                for row in response.json()["results"]:
                    scores[int(row["index"])] = float(row["relevance_score"])
                if any(score is None for score in scores):
                    raise RuntimeError("OpenRouter reranker returned incomplete scores")
                return [float(score) for score in scores if score is not None]
            if response.status_code not in {408, 409, 429, 500, 502, 503, 504, 524, 529} or attempt == self.max_retries:
                raise RuntimeError(
                    f"OpenRouter rerank request failed with HTTP {response.status_code}: {response.text[:500]}"
                )
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(60, 2**attempt)
            except ValueError:
                delay = min(60, 2**attempt)
            sleep(max(0, delay))
        raise AssertionError("unreachable")

    def close(self) -> None:
        """Requests do not hold process-level resources."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "openrouter-rerank",
            "model": self.model,
            "request_concurrency": self.request_concurrency,
        }
