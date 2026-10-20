"""Sentence Transformers multi-vector embedding component."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from modern_ir_bench.embeddings.multi_vector_cache import SqliteMultiVectorCache


class SentenceTransformersMultiVectorEmbedding:
    """Encode text or images into token matrices for late interaction."""

    def __init__(
        self,
        *,
        model: str,
        revision: str | None = None,
        dimension: int | None = None,
        batch_size: int = 16,
        device: str | None = None,
        normalize: bool = True,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_path: Path | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        config_kwargs: Mapping[str, Any] | None = None,
        encode_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_id = model
        self.revision = revision
        self._dimension = dimension
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = dict(model_kwargs or {})
        self.config_kwargs = dict(config_kwargs or {})
        self.encode_kwargs = dict(encode_kwargs or {})
        self.cache = SqliteMultiVectorCache(cache_path)
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import MultiVectorEncoder

            self._model = MultiVectorEncoder(
                self.model_id,
                revision=self.revision,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                model_kwargs=self.model_kwargs,
                config_kwargs=self.config_kwargs,
                similarity_fn_name="maxsim",
            )
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            dimension = self._load_model().get_embedding_dimension()
            if dimension is None:
                raise ValueError(f"Could not determine embedding dimension for {self.model_id}")
            self._dimension = int(dimension)
        return self._dimension

    def _encode(self, inputs: Sequence[Any], *, role: str) -> list[np.ndarray]:
        values = list(inputs)
        keys = [self._cache_key(value, role=role) for value in values]
        cached = self.cache.read(keys)
        missing: dict[str, Any] = {}
        for key, value in zip(keys, values, strict=True):
            missing.setdefault(key, value)

        additions: dict[str, np.ndarray] = {}
        missing_keys = [key for key in missing if key not in cached]
        if missing_keys:
            model = self._load_model()
            method = model.encode_document if role == "document" else model.encode_query
            encoded = method(
                [missing[key] for key in missing_keys],
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                **self.encode_kwargs,
            )
            for key, value in zip(missing_keys, encoded, strict=True):
                additions[key] = self._validate(value)
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
            raise ValueError(
                f"Expected a non-empty token matrix with dimension {self.dimension}, got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("multi-vector embedding contains a non-finite value")
        return matrix

    def _cache_key(self, value: Any, *, role: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_id.encode())
        digest.update(b"\0")
        digest.update((self.revision or "default").encode())
        digest.update(b"\0")
        digest.update(role.encode())
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
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
