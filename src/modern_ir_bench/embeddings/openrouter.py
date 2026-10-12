"""Cached embeddings through OpenRouter's OpenAI-compatible API."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import sleep
from typing import Any, Literal

import numpy as np

from modern_ir_bench.embeddings.sqlite_cache import SqliteVectorCache


class OpenRouterEmbedding:
    """Embed text with distinct document/query input types and a durable cache."""

    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        batch_size: int = 64,
        document_input_type: str = "document",
        query_input_type: str = "query",
        cache_path: Path | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        timeout_seconds: float = 300,
        max_retries: int = 6,
        request_concurrency: int = 4,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        if dimension < 1 or batch_size < 1:
            raise ValueError("dimension and batch size must be positive")
        if timeout_seconds <= 0 or max_retries < 0 or request_concurrency < 1:
            raise ValueError("timeout and concurrency must be positive and retries non-negative")
        self.model_id = model
        self._dimension = dimension
        self.batch_size = batch_size
        self.document_input_type = document_input_type
        self.query_input_type = query_input_type
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.request_concurrency = request_concurrency
        self.cache = SqliteVectorCache(cache_path)

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(inputs, kind="document")

    def encode_queries(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(inputs, kind="query")

    def _encode(
        self,
        inputs: Sequence[Any],
        *,
        kind: Literal["document", "query"],
    ) -> np.ndarray:
        texts = list(inputs)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("OpenRouterEmbedding only accepts strings")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        input_type = self.document_input_type if kind == "document" else self.query_input_type
        keys = [self._cache_key(text, input_type) for text in texts]
        cached = self.cache.read(keys)
        missing: list[tuple[str, str]] = []
        seen: set[str] = set()
        for key, text in zip(keys, texts, strict=True):
            if key not in cached and key not in seen:
                missing.append((key, text))
                seen.add(key)

        batches = [
            missing[start : start + self.batch_size]
            for start in range(0, len(missing), self.batch_size)
        ]
        with ThreadPoolExecutor(max_workers=self.request_concurrency) as executor:
            requests = [
                executor.submit(self._request, [text for _, text in batch], input_type)
                for batch in batches
            ]
            for batch, request in zip(batches, requests, strict=True):
                vectors = request.result()
                values = {
                    key: self._validate_vector(vector)
                    for (key, _), vector in zip(batch, vectors, strict=True)
                }
                cached.update(values)
                self.cache.write(values)
        return np.asarray([cached[key] for key in keys], dtype=np.float32)

    def _request(self, texts: list[str], input_type: str) -> list[list[float]]:
        import requests

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for {self.model_id}")
        payload = {
            "model": self.model_id,
            "input": texts,
            "input_type": input_type,
            "dimensions": self.dimension,
            "encoding_format": "float",
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                sleep(min(60, 2**attempt))
                continue
            if response.status_code == 200:
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                return [item["embedding"] for item in data]
            if response.status_code not in {408, 409, 429, 500, 502, 503, 504, 529} or attempt == self.max_retries:
                raise RuntimeError(
                    f"OpenRouter embeddings request failed with HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(60, 2**attempt)
            except ValueError:
                delay = min(60, 2**attempt)
            sleep(max(0, delay))
        raise AssertionError("unreachable")

    def _validate_vector(self, vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        if len(values) != self.dimension:
            raise ValueError(f"OpenRouter returned {len(values)} dimensions; expected {self.dimension}")
        return values

    def _cache_key(self, text: str, input_type: str) -> str:
        value = f"openrouter\0{self.model_id}\0{self.dimension}\0{input_type}\0{text}"
        return sha256(value.encode()).hexdigest()
