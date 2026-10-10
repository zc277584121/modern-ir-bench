"""Cached, token-aware Voyage dense embedding component."""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterable, Sequence
from hashlib import sha256
from pathlib import Path
from time import sleep
from typing import Any, Literal

import numpy as np


class VoyageEmbedding:
    def __init__(
        self,
        *,
        model: str = "voyage-4-large",
        dimension: int = 1024,
        batch_size: int = 128,
        max_batch_tokens: int = 100_000,
        cache_path: Path | None = None,
        max_rate_limit_retries: int = 5,
    ) -> None:
        if dimension < 1 or batch_size < 1 or max_batch_tokens < 1:
            raise ValueError("dimension, batch size, and token limit must be positive")
        if max_rate_limit_retries < 0:
            raise ValueError("max_rate_limit_retries must be non-negative")
        self.model_id = model
        self._dimension = dimension
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.cache_path = cache_path
        self.max_rate_limit_retries = max_rate_limit_retries
        self._client: Any | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self) -> Any:
        if self._client is None:
            import voyageai

            self._client = voyageai.Client()
        return self._client

    def _encode(
        self,
        inputs: Sequence[Any],
        *,
        kind: Literal["document", "query"],
    ) -> np.ndarray:
        texts = [str(value) for value in inputs]
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        keys = [self._cache_key(text, kind) for text in texts]
        cached = self._read_cache(keys)
        missing = list(dict.fromkeys((key, text) for key, text in zip(keys, texts, strict=True) if key not in cached))
        if missing:
            client = self._get_client()
            encodings = client.tokenize([text for _, text in missing], model=self.model_id)
            for batch in self._batches(missing, encodings):
                response = self._embed_with_retry([text for _, text in batch], kind)
                values = {
                    key: self._validate_vector(vector)
                    for (key, _), vector in zip(batch, response.embeddings, strict=True)
                }
                cached.update(values)
                self._write_cache(values)
        return np.asarray([cached[key] for key in keys], dtype=np.float32)

    def _validate_vector(self, vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        if len(values) != self.dimension:
            raise ValueError(f"Voyage returned {len(values)} dimensions; expected {self.dimension}")
        return values

    def _batches(
        self,
        items: Sequence[tuple[str, str]],
        encodings: Sequence[Any],
    ) -> Iterable[list[tuple[str, str]]]:
        batch: list[tuple[str, str]] = []
        token_count = 0
        for item, encoding in zip(items, encodings, strict=True):
            item_tokens = len(encoding.ids)
            if item_tokens > self.max_batch_tokens:
                raise ValueError(
                    f"One input contains {item_tokens} tokens; the configured batch limit is {self.max_batch_tokens}"
                )
            if batch and (len(batch) >= self.batch_size or token_count + item_tokens > self.max_batch_tokens):
                yield batch
                batch = []
                token_count = 0
            batch.append(item)
            token_count += item_tokens
        if batch:
            yield batch

    def _embed_with_retry(self, texts: list[str], kind: str) -> Any:
        from voyageai.error import RateLimitError

        for attempt in range(self.max_rate_limit_retries + 1):
            try:
                return self._get_client().embed(
                    texts,
                    model=self.model_id,
                    input_type=kind,
                    output_dimension=self.dimension,
                )
            except RateLimitError:
                if attempt == self.max_rate_limit_retries:
                    raise
                sleep(min(120, 10 * 2**attempt))
        raise AssertionError("unreachable")

    def _cache_key(self, text: str, kind: str) -> str:
        return sha256(f"{self.model_id}\0{self.dimension}\0{kind}\0{text}".encode()).hexdigest()

    def _read_cache(self, keys: Sequence[str]) -> dict[str, list[float]]:
        if self.cache_path is None or not self.cache_path.exists() or not keys:
            return {}
        found: dict[str, list[float]] = {}
        with sqlite3.connect(self.cache_path) as connection:
            self._create_cache(connection)
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                rows = connection.execute(
                    f"SELECT cache_key, dimension, vector FROM embeddings "
                    f"WHERE cache_key IN ({','.join('?' for _ in batch)})",
                    list(batch),
                )
                found.update({key: list(struct.unpack(f"<{dimension}f", blob)) for key, dimension, blob in rows})
        return found

    def _write_cache(self, values: dict[str, list[float]]) -> None:
        if self.cache_path is None or not values:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [(key, len(vector), struct.pack(f"<{len(vector)}f", *vector)) for key, vector in values.items()]
        with sqlite3.connect(self.cache_path) as connection:
            self._create_cache(connection)
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings(cache_key, dimension, vector) VALUES (?, ?, ?)",
                rows,
            )

    @staticmethod
    def _create_cache(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, dimension INTEGER, vector BLOB)"
        )

    def encode_documents(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(inputs, kind="document")

    def encode_queries(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(inputs, kind="query")
