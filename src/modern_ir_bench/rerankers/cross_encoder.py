"""Pinned local CrossEncoder reranker."""

from __future__ import annotations

import atexit
from collections.abc import Mapping, Sequence
from typing import Any

from modern_ir_bench.rerankers.protocols import validate_batch


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model: str,
        revision: str,
        max_length: int = 4096,
        batch_size: int = 2,
        devices: tuple[str, ...] | None = None,
        use_fp16: bool = True,
    ) -> None:
        if not model or not revision:
            raise ValueError("Model and revision must be pinned")
        if max_length < 1 or batch_size < 1:
            raise ValueError("Maximum length and batch size must be positive")
        self.model = model
        self.revision = revision
        self.id = f"cross-encoder:{model}@{revision}"
        self.max_length = max_length
        self.batch_size = batch_size
        self.devices = devices
        self.use_fp16 = use_fp16
        self._encoder: Any | None = None
        self._pool: Any | None = None
        atexit.register(self.close)

    def score_batch(
        self,
        queries: Sequence[str],
        documents: Sequence[Sequence[str]],
    ) -> list[list[float]]:
        validate_batch(queries, documents)
        counts = [len(rows) for rows in documents]
        pairs = [[query, document] for query, rows in zip(queries, documents) for document in rows]
        if not pairs:
            return [[] for _ in queries]
        encoder = self._load()
        scores = encoder.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            pool=self._pool,
        )
        flat = [float(score) for score in scores]
        grouped = []
        offset = 0
        for count in counts:
            grouped.append(flat[offset : offset + count])
            offset += count
        return grouped

    def _load(self) -> Any:
        if self._encoder is None:
            import torch
            from sentence_transformers import CrossEncoder

            model_kwargs = {"torch_dtype": torch.float16} if self.use_fp16 else None
            primary_device = self.devices[0] if self.devices else None
            self._encoder = CrossEncoder(
                self.model,
                revision=self.revision,
                max_length=self.max_length,
                device=primary_device,
                model_kwargs=model_kwargs,
            )
        if self.devices and len(self.devices) > 1 and self._pool is None:
            self._pool = self._encoder.start_multi_process_pool(target_devices=list(self.devices))
        return self._encoder

    def close(self) -> None:
        if self._pool is not None and self._encoder is not None:
            self._encoder.stop_multi_process_pool(self._pool)
            self._pool = None

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "cross-encoder",
            "model": self.model,
            "revision": self.revision,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "devices": list(self.devices) if self.devices else None,
            "precision": "float16" if self.use_fp16 else "float32",
        }
