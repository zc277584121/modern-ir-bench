"""Voyage managed reranker with bounded concurrent requests."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from typing import Any

from modern_ir_bench.rerankers.protocols import validate_batch


class VoyageReranker:
    def __init__(
        self,
        *,
        model: str = "rerank-3",
        request_concurrency: int = 8,
        max_retries: int = 6,
    ) -> None:
        if not model or request_concurrency < 1 or max_retries < 0:
            raise ValueError("Model and concurrency must be positive and retries non-negative")
        self.model = model
        self.id = f"voyage:{model}"
        self.request_concurrency = request_concurrency
        self.max_retries = max_retries
        self._thread_state = threading.local()

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
        if not documents:
            return []
        retryable = self._retryable_errors()
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client().rerank(
                    query,
                    documents,
                    model=self.model,
                    top_k=None,
                    truncation=True,
                )
                scores: list[float | None] = [None] * len(documents)
                for result in response.results:
                    scores[int(result.index)] = float(result.relevance_score)
                if any(score is None for score in scores):
                    raise RuntimeError("Voyage reranker returned incomplete scores")
                return [float(score) for score in scores if score is not None]
            except retryable:
                if attempt == self.max_retries:
                    raise
                sleep(min(60, 2**attempt))
        raise AssertionError("unreachable")

    def _client(self) -> Any:
        client = getattr(self._thread_state, "client", None)
        if client is None:
            import voyageai

            client = voyageai.Client()
            self._thread_state.client = client
        return client

    @staticmethod
    def _retryable_errors() -> tuple[type[BaseException], ...]:
        from voyageai.error import APIConnectionError, RateLimitError, ServerError, ServiceUnavailableError

        return APIConnectionError, RateLimitError, ServerError, ServiceUnavailableError

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "voyage-rerank",
            "model": self.model,
            "request_concurrency": self.request_concurrency,
            "truncation": True,
        }
