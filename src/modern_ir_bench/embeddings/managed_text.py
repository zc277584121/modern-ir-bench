"""Cached text embeddings from Jina and Alibaba Cloud Model Studio."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import sleep
from typing import Any, Literal

import numpy as np

from modern_ir_bench.embeddings.sqlite_cache import SqliteVectorCache

InputKind = Literal["document", "query"]


class ManagedTextEmbedding(ABC):
    """Shared batching, retries, validation, and caching for hosted text models."""

    backend: str

    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        batch_size: int,
        cache_path: Path | None,
        api_key_env: str,
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
        self.cache = SqliteVectorCache(cache_path)
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.request_concurrency = request_concurrency

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(inputs, kind="document")

    def encode_queries(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(inputs, kind="query")

    def _encode(self, inputs: Sequence[Any], *, kind: InputKind) -> np.ndarray:
        texts = list(inputs)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError(f"{type(self).__name__} accepts text strings")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        keys = [self._cache_key(text, kind) for text in texts]
        cached = self.cache.read(keys)
        missing: list[tuple[str, str]] = []
        seen: set[str] = set()
        for key, text in zip(keys, texts, strict=True):
            if key not in cached and key not in seen:
                missing.append((key, text))
                seen.add(key)

        batches = [missing[start : start + self.batch_size] for start in range(0, len(missing), self.batch_size)]
        with ThreadPoolExecutor(max_workers=self.request_concurrency) as executor:
            requests = [
                executor.submit(self._request_with_retries, [text for _, text in batch], kind)
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

    def _request_with_retries(self, texts: list[str], kind: InputKind) -> list[list[float]]:
        import requests

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for {self.model_id}")
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    payload=self._payload(texts, kind),
                )
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                sleep(min(60, 2**attempt))
                continue
            if response.status_code == 200:
                return self._vectors(response.json())
            if response.status_code not in {408, 409, 429, 500, 502, 503, 504, 529} or attempt == self.max_retries:
                raise RuntimeError(
                    f"{self.backend} embeddings request failed with HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(60, 2**attempt)
            except ValueError:
                delay = min(60, 2**attempt)
            sleep(max(0, delay))
        raise AssertionError("unreachable")

    def _post(self, *, headers: dict[str, str], payload: dict[str, object]):
        import requests

        return requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )

    def _cache_key(self, text: str, kind: InputKind) -> str:
        digest = sha256()
        digest.update(f"{self.backend}\0{self.model_id}\0{self.dimension}\0{kind}\0".encode())
        digest.update(text.encode())
        return digest.hexdigest()

    def _validate_vector(self, vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        if len(values) != self.dimension:
            raise ValueError(f"{self.backend} returned {len(values)} dimensions; expected {self.dimension}")
        return values

    @property
    @abstractmethod
    def endpoint(self) -> str: ...

    @abstractmethod
    def _payload(self, texts: list[str], kind: InputKind) -> dict[str, object]: ...

    @abstractmethod
    def _vectors(self, response: dict[str, Any]) -> list[list[float]]: ...


class JinaTextEmbedding(ManagedTextEmbedding):
    """Jina text embeddings with retrieval-specific query and passage tasks."""

    backend = "jina"

    def __init__(
        self,
        *,
        model: str = "jina-embeddings-v5-text-small",
        dimension: int = 1024,
        batch_size: int = 32,
        cache_path: Path | None = None,
        resolve_ip: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dimension=dimension,
            batch_size=batch_size,
            cache_path=cache_path,
            api_key_env="JINA_API_KEY",
            **kwargs,
        )
        self.resolve_ip = resolve_ip

    @property
    def endpoint(self) -> str:
        return "https://api.jina.ai/v1/embeddings"

    def _payload(self, texts: list[str], kind: InputKind) -> dict[str, object]:
        return {
            "model": self.model_id,
            "input": texts,
            "task": "retrieval.passage" if kind == "document" else "retrieval.query",
            "dimensions": self.dimension,
            "embedding_type": "float",
        }

    def _vectors(self, response: dict[str, Any]) -> list[list[float]]:
        rows = sorted(response["data"], key=lambda item: item["index"])
        return [row["embedding"] for row in rows]

    def _post(self, *, headers: dict[str, str], payload: dict[str, object]):
        if not self.resolve_ip:
            return super()._post(headers=headers, payload=payload)

        import requests
        from requests.adapters import HTTPAdapter

        hostname = "api.jina.ai"
        resolved_ip = self.resolve_ip

        class StaticDnsAdapter(HTTPAdapter):
            def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
                _, pool_kwargs = self.build_connection_pool_key_attributes(request, verify, cert)
                pool_kwargs.update(assert_hostname=hostname, server_hostname=hostname)
                return self.poolmanager.connection_from_host(
                    resolved_ip,
                    port=443,
                    scheme="https",
                    pool_kwargs=pool_kwargs,
                )

        with requests.Session() as session:
            session.mount(f"https://{hostname}/", StaticDnsAdapter())
            resolved_headers = {**headers, "Host": hostname}
            return session.post(
                self.endpoint,
                headers=resolved_headers,
                json=payload,
                timeout=self.timeout_seconds,
            )


class DashScopeTextEmbedding(ManagedTextEmbedding):
    """DashScope text embeddings with native query/document asymmetry."""

    backend = "dashscope"

    def __init__(
        self,
        *,
        model: str = "text-embedding-v4",
        dimension: int = 1024,
        batch_size: int = 10,
        cache_path: Path | None = None,
        query_instruction: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dimension=dimension,
            batch_size=batch_size,
            cache_path=cache_path,
            api_key_env="DASHSCOPE_API_KEY",
            **kwargs,
        )
        self.query_instruction = query_instruction

    @property
    def endpoint(self) -> str:
        return "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

    def _payload(self, texts: list[str], kind: InputKind) -> dict[str, object]:
        parameters: dict[str, object] = {
            "dimension": self.dimension,
            "output_type": "dense",
            "text_type": kind,
        }
        if kind == "query" and self.query_instruction:
            parameters["instruct"] = self.query_instruction
        return {"model": self.model_id, "input": {"texts": texts}, "parameters": parameters}

    def _vectors(self, response: dict[str, Any]) -> list[list[float]]:
        rows = sorted(response["output"]["embeddings"], key=lambda item: item["text_index"])
        return [row["embedding"] for row in rows]

    def _cache_key(self, text: str, kind: InputKind) -> str:
        digest = sha256()
        digest.update(
            f"{self.backend}\0{self.model_id}\0{self.dimension}\0{kind}\0{self.query_instruction or ''}\0".encode()
        )
        digest.update(text.encode())
        return digest.hexdigest()
