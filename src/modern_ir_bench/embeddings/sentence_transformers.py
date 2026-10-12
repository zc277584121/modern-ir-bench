"""Sentence Transformers dense embedding component."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np


class SentenceTransformersEmbedding:
    """Use one pinned Sentence Transformers model for documents and queries."""

    def __init__(
        self,
        *,
        model: str,
        revision: str | None = None,
        batch_size: int = 128,
        device: str | None = None,
        normalize: bool = True,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        document_prompt: str | None = None,
        query_prompt: str | None = None,
        document_prompt_name: str | None = None,
        query_prompt_name: str | None = None,
        dimension: int | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        config_kwargs: Mapping[str, Any] | None = None,
        encode_kwargs: Mapping[str, Any] | None = None,
        document_method: Literal["encode", "encode_document"] = "encode",
        query_method: Literal["encode", "encode_query"] = "encode",
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if document_prompt is not None and document_prompt_name is not None:
            raise ValueError("document_prompt and document_prompt_name are mutually exclusive")
        if query_prompt is not None and query_prompt_name is not None:
            raise ValueError("query_prompt and query_prompt_name are mutually exclusive")

        self.model_id = model
        self.revision = revision
        self.batch_size = batch_size
        self.normalize = normalize
        self.document_prompt = document_prompt
        self.query_prompt = query_prompt
        self.document_prompt_name = document_prompt_name
        self.query_prompt_name = query_prompt_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.model_kwargs = dict(model_kwargs or {})
        self.config_kwargs = dict(config_kwargs or {})
        self.encode_kwargs = dict(encode_kwargs or {})
        self.document_method = document_method
        self.query_method = query_method
        self._model: Any | None = None
        self._dimension = dimension

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.model_id,
            revision=self.revision,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            model_kwargs=self.model_kwargs,
            config_kwargs=self.config_kwargs,
        )
        return self._model

    def _model_dimension(self) -> int:
        model = self._load_model()
        get_dimension = getattr(model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = model.get_sentence_embedding_dimension
        dimension = get_dimension()
        if dimension is None:
            raise ValueError(f"Could not determine embedding dimension for {self.model_id}")
        return int(dimension)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self._model_dimension()
        return self._dimension

    def _encode(
        self,
        inputs: Sequence[Any],
        *,
        prompt: str | None,
        prompt_name: str | None,
        method: Literal["encode", "encode_document", "encode_query"],
    ) -> np.ndarray:
        values = list(inputs)
        prompt_arguments = {}
        if prompt is not None:
            prompt_arguments["prompt"] = prompt
        if prompt_name is not None:
            prompt_arguments["prompt_name"] = prompt_name
        return np.asarray(
            getattr(self._load_model(), method)(
                values,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
                **self.encode_kwargs,
                **prompt_arguments,
            ),
            dtype=np.float32,
        )

    def encode_documents(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(
            inputs,
            prompt=self.document_prompt,
            prompt_name=self.document_prompt_name,
            method=self.document_method,
        )

    def encode_queries(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(
            inputs,
            prompt=self.query_prompt,
            prompt_name=self.query_prompt_name,
            method=self.query_method,
        )

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
