"""Sparse embedding protocol used by learned lexical retrieval Solutions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

SparseVector = dict[int, float]


class SparseEmbedding(Protocol):
    def encode_sparse_documents(self, inputs: Sequence[Any]) -> list[SparseVector]: ...

    def encode_sparse_queries(self, inputs: Sequence[Any]) -> list[SparseVector]: ...
