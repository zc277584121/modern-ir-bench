"""Pinned BGE-M3 dense and learned-sparse embedding component."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from modern_ir_bench.embeddings.sparse import SparseVector


class BgeM3Embedding:
    """Share one BGE-M3 model and cache between dense and sparse Solutions."""

    dimension = 1024

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-m3",
        revision: str,
        batch_size: int = 12,
        device: str | None = None,
        use_fp16: bool = True,
        query_max_length: int = 512,
        passage_max_length: int = 1024,
    ) -> None:
        if not revision:
            raise ValueError("BGE-M3 revision must be pinned")
        self.model_id = model
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self._model: Any | None = None
        self._cache: dict[tuple[str, str], tuple[np.ndarray, SparseVector]] = {}

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from FlagEmbedding import BGEM3FlagModel
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(repo_id=self.model_id, revision=self.revision)
        devices = [self.device] if self.device else None
        self._model = BGEM3FlagModel(
            model_path,
            devices=devices,
            use_fp16=self.use_fp16,
            query_max_length=self.query_max_length,
            passage_max_length=self.passage_max_length,
        )
        return self._model

    def _encode(
        self,
        inputs: Sequence[Any],
        *,
        kind: Literal["document", "query"],
    ) -> list[tuple[np.ndarray, SparseVector]]:
        texts = [str(value) for value in inputs]
        missing = [text for text in texts if (kind, text) not in self._cache]
        if missing:
            model = self._load_model()
            encode = model.encode_queries if kind == "query" else model.encode_corpus
            output = encode(
                missing,
                batch_size=self.batch_size,
                max_length=(self.query_max_length if kind == "query" else self.passage_max_length),
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense = np.asarray(output["dense_vecs"], dtype=np.float32)
            sparse = [
                {int(token_id): float(weight) for token_id, weight in weights.items()}
                for weights in output["lexical_weights"]
            ]
            self._cache.update(
                {
                    (kind, text): (dense_vector, sparse_vector)
                    for text, dense_vector, sparse_vector in zip(
                        missing,
                        dense,
                        sparse,
                        strict=True,
                    )
                }
            )
        return [self._cache[(kind, text)] for text in texts]

    def encode_documents(self, inputs: Sequence[Any]) -> np.ndarray:
        return np.stack([dense for dense, _ in self._encode(inputs, kind="document")])

    def encode_queries(self, inputs: Sequence[Any]) -> np.ndarray:
        return np.stack([dense for dense, _ in self._encode(inputs, kind="query")])

    def encode_sparse_documents(self, inputs: Sequence[Any]) -> list[SparseVector]:
        return [sparse for _, sparse in self._encode(inputs, kind="document")]

    def encode_sparse_queries(self, inputs: Sequence[Any]) -> list[SparseVector]:
        return [sparse for _, sparse in self._encode(inputs, kind="query")]

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
