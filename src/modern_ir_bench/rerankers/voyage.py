"""Voyage managed reranker with bounded concurrent requests."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from time import monotonic, sleep
from typing import Any

from modern_ir_bench.rerankers.protocols import validate_batch


class VoyageReranker:
    def __init__(
        self,
        *,
        model: str = "rerank-3",
        request_concurrency: int = 4,
        request_interval_seconds: float = 1.0,
        rate_limit_cooldown_seconds: float = 65.0,
        max_retries: int = 6,
    ) -> None:
        if not model or request_concurrency < 1 or request_interval_seconds < 0 or rate_limit_cooldown_seconds <= 0:
            raise ValueError("Model, concurrency, request interval, and rate-limit cooldown must be valid")
        if max_retries < 0:
            raise ValueError("Retries must be non-negative")
        self.model = model
        self.id = f"voyage:{model}"
        self.request_concurrency = request_concurrency
        self.request_interval_seconds = request_interval_seconds
        self.rate_limit_cooldown_seconds = rate_limit_cooldown_seconds
        self.max_retries = max_retries
        self._thread_state = threading.local()
        self._rate_condition = threading.Condition()
        self._next_request_at = 0.0

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
        rate_limit_error, retryable = self._retryable_errors()
        for attempt in range(self.max_retries + 1):
            try:
                self._wait_for_request_slot()
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
            except rate_limit_error:
                if attempt == self.max_retries:
                    raise
                self._cool_down()
            except retryable:
                if attempt == self.max_retries:
                    raise
                sleep(min(60, 2**attempt))
        raise AssertionError("unreachable")

    def _wait_for_request_slot(self) -> None:
        with self._rate_condition:
            while True:
                delay = self._next_request_at - monotonic()
                if delay <= 0:
                    self._next_request_at = monotonic() + self.request_interval_seconds
                    return
                self._rate_condition.wait(timeout=delay)

    def _cool_down(self) -> None:
        with self._rate_condition:
            self._next_request_at = max(
                self._next_request_at,
                monotonic() + self.rate_limit_cooldown_seconds,
            )
            self._rate_condition.notify_all()

    def _client(self) -> Any:
        client = getattr(self._thread_state, "client", None)
        if client is None:
            import voyageai

            client = voyageai.Client()
            self._thread_state.client = client
        return client

    @staticmethod
    def _retryable_errors() -> tuple[type[BaseException], tuple[type[BaseException], ...]]:
        from voyageai.error import APIConnectionError, RateLimitError, ServerError, ServiceUnavailableError

        return RateLimitError, (APIConnectionError, ServerError, ServiceUnavailableError)

    def close(self) -> None:
        """Voyage clients do not hold process-level resources."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "voyage-rerank",
            "model": self.model,
            "request_concurrency": self.request_concurrency,
            "request_interval_seconds": self.request_interval_seconds,
            "rate_limit_cooldown_seconds": self.rate_limit_cooldown_seconds,
            "truncation": True,
        }
