"""Sentence Transformers multi-vector embedding component."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from modern_ir_bench.embeddings.cached_multi_vector import CachedMultiVectorEmbedding
from modern_ir_bench.embeddings.model_snapshot import pinned_adapter_snapshot


class SentenceTransformersMultiVectorEmbedding(CachedMultiVectorEmbedding):
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
        normalize_float32: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_path: Path | None = None,
        cache_dir: Path | None = None,
        cache_only: bool = False,
        base_revision: str | None = None,
        document_length: int | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        config_kwargs: Mapping[str, Any] | None = None,
        encode_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if document_length is not None and document_length < 1:
            raise ValueError("document_length must be positive")
        if base_revision is not None and revision is None:
            raise ValueError("A pinned adapter backbone requires a pinned adapter revision")
        self.base_revision = base_revision
        self.normalize = normalize
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.document_length = document_length
        self.model_kwargs = dict(model_kwargs or {})
        self.config_kwargs = dict(config_kwargs or {})
        self.encode_kwargs = dict(encode_kwargs or {})
        super().__init__(
            model=model,
            revision=revision,
            dimension=dimension,
            batch_size=batch_size,
            device=device,
            cache_path=cache_path,
            cache_dir=cache_dir,
            cache_only=cache_only,
            normalize_float32=normalize_float32,
            configuration={
                "base_revision": base_revision,
                "normalize": normalize,
                "document_length": document_length,
                "model_kwargs": self.model_kwargs,
                "config_kwargs": self.config_kwargs,
                "encode_kwargs": self.encode_kwargs,
                "trust_remote_code": trust_remote_code,
            },
        )

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import MultiVectorEncoder

            model_path = self.model_id
            if self.base_revision is not None:
                model_path = str(
                    pinned_adapter_snapshot(
                        self.model_id, self.revision, self.base_revision, local_files_only=self.local_files_only
                    )
                )
            self._model = MultiVectorEncoder(
                model_path,
                revision=self.revision if self.base_revision is None else None,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                model_kwargs=self.model_kwargs,
                config_kwargs=self.config_kwargs,
                similarity_fn_name="maxsim",
            )
            if self.document_length is not None:
                self._model[0].document_length = self.document_length
        return self._model

    def _encode_missing(self, inputs: Sequence[Any], *, role: str) -> Sequence[Any]:
        model = self._load_model()
        method = model.encode_document if role == "document" else model.encode_query
        return method(
            [str(value) if isinstance(value, Path) else value for value in inputs],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            **self.encode_kwargs,
        )
