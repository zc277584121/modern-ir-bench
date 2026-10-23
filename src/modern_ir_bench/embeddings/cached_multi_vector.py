"""Persistent token matrices and an explicit cache-only inference boundary."""

from __future__ import annotations

import gc
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from modern_ir_bench.embeddings.multi_vector_cache import SqliteMultiVectorCache


class CachedMultiVectorEmbedding(ABC):
    """Share cache validation and replay between native multi-vector encoders."""

    def __init__(
        self,
        *,
        model: str,
        revision: str | None,
        dimension: int | None,
        batch_size: int,
        device: str | None,
        configuration: Mapping[str, Any],
        cache_path: Path | None = None,
        cache_dir: Path | None = None,
        cache_only: bool = False,
        normalize_float32: bool = False,
    ) -> None:
        if not model or batch_size < 1 or (dimension is not None and dimension < 1):
            raise ValueError("model, batch_size and dimension must be valid")
        if cache_path is not None and cache_dir is not None:
            raise ValueError("Choose cache_path or a configuration-addressed cache_dir")
        self.model_id = model
        self.revision = revision
        self._dimension = dimension
        self.batch_size = batch_size
        self.device = device
        self.cache_only = cache_only
        self.normalize_float32 = normalize_float32
        self.configuration = {
            "encoder": type(self).__name__,
            "model": model,
            "revision": revision,
            "dimension": dimension,
            "normalize_float32": normalize_float32,
            **configuration,
        }
        serialized = json.dumps(self.configuration, sort_keys=True, separators=(",", ":"), default=str)
        self.cache_fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"{model.replace('/', '--')}-{self.cache_fingerprint[:16]}.sqlite3"
        self.cache = SqliteMultiVectorCache(cache_path)
        self._model: Any | None = None

    @abstractmethod
    def _load_model(self) -> Any: ...

    @abstractmethod
    def _encode_missing(self, inputs: Sequence[Any], *, role: str) -> Sequence[Any]: ...

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            if self.cache_only:
                raise ValueError("Cache-only encoding requires an explicit dimension")
            dimension = self._load_model().get_embedding_dimension()
            if dimension is None:
                raise ValueError(f"Could not determine embedding dimension for {self.model_id}")
            self._dimension = int(dimension)
        return self._dimension

    def _encode(self, inputs: Sequence[Any], *, role: str) -> list[np.ndarray]:
        values = list(inputs)
        keys = [self._cache_key(value, role=role) for value in values]
        cached = self.cache.read(keys)
        missing = {key: value for key, value in zip(keys, values, strict=True) if key not in cached}
        if missing and self.cache_only:
            raise ValueError(
                f"Missing {len(missing)} {role} embeddings in {self.cache.path}; run the prepare stage first"
            )
        # Commit each batch so a later failure never discards completed encoding.
        items = list(missing.items())
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            encoded = self._encode_missing([value for _, value in batch], role=role)
            additions = {key: self._prepare_output(value) for (key, _), value in zip(batch, encoded, strict=True)}
            self.cache.write(additions)
            cached.update(additions)
        return [self._validate(cached[key]) for key in keys]

    def encode_documents(self, inputs: Sequence[Any]) -> list[np.ndarray]:
        return self._encode(inputs, role="document")

    def encode_queries(self, inputs: Sequence[Any]) -> list[np.ndarray]:
        return self._encode(inputs, role="query")

    def _validate(self, value: Any) -> np.ndarray:
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != self.dimension:
            raise ValueError(f"Expected a non-empty token matrix with dimension {self.dimension}, got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("multi-vector embedding contains a non-finite value")
        return matrix

    def _prepare_output(self, value: Any) -> np.ndarray:
        matrix = self._validate(value)
        if self.normalize_float32:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms == 0).any():
                raise ValueError("Cannot normalize zero or overflowing token vectors")
            matrix = matrix / norms
        return matrix

    def _cache_key(self, value: Any, *, role: str) -> str:
        digest = hashlib.sha256()
        for part in (self.model_id, self.revision or "default", role):
            digest.update(part.encode())
            digest.update(b"\0")
        if isinstance(value, Path):
            with value.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
        elif isinstance(value, bytes):
            digest.update(value)
        else:
            digest.update(str(value).encode("utf-8"))
        return digest.hexdigest()

    def release(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
