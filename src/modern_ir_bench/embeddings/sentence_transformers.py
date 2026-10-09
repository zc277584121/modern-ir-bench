"""Sentence Transformers dense embedding component."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class SentenceTransformersEmbedding:
    """Use one fixed Sentence Transformers model for documents and queries."""

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
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if document_prompt is not None and document_prompt_name is not None:
            raise ValueError("document_prompt and document_prompt_name are mutually exclusive")
        if query_prompt is not None and query_prompt_name is not None:
            raise ValueError("query_prompt and query_prompt_name are mutually exclusive")

        from sentence_transformers import SentenceTransformer

        self.model_id = model
        self.revision = revision
        self.batch_size = batch_size
        self.normalize = normalize
        self.document_prompt = document_prompt
        self.query_prompt = query_prompt
        self.document_prompt_name = document_prompt_name
        self.query_prompt_name = query_prompt_name
        self._model = SentenceTransformer(
            model,
            revision=revision,
            device=device,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = self._model.get_sentence_embedding_dimension
        dimension = get_dimension()
        if dimension is None:
            raise ValueError(f"Could not determine embedding dimension for {model}")
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(
        self,
        inputs: Sequence[Any],
        *,
        prompt: str | None,
        prompt_name: str | None,
    ) -> np.ndarray:
        texts = list(inputs)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("SentenceTransformersEmbedding only accepts strings")
        prompt_arguments = {}
        if prompt is not None:
            prompt_arguments["prompt"] = prompt
        if prompt_name is not None:
            prompt_arguments["prompt_name"] = prompt_name
        return np.asarray(
            self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
                **prompt_arguments,
            ),
            dtype=np.float32,
        )

    def encode_documents(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(
            inputs,
            prompt=self.document_prompt,
            prompt_name=self.document_prompt_name,
        )

    def encode_queries(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(
            inputs,
            prompt=self.query_prompt,
            prompt_name=self.query_prompt_name,
        )
