"""Lifecycle protocol implemented by dense index backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from modern_ir_bench.embeddings.protocols import DenseVectors, MultiVectors
from modern_ir_bench.retrieval.types import SearchHit


class DenseIndexSession(Protocol):
    """Mutable index state owned by one Solution evaluation."""

    def add(self, ids: Sequence[str], vectors: DenseVectors) -> None: ...

    def seal(self) -> None: ...

    def search(
        self,
        query_vectors: DenseVectors,
        *,
        top_k: int,
    ) -> list[list[SearchHit]]: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class DenseIndex(Protocol):
    """Factory for an isolated dense index session."""

    def open(self, *, dimension: int) -> DenseIndexSession: ...


class MultiVectorIndexSession(Protocol):
    """Mutable index state for variable-length token matrices."""

    def add(self, ids: Sequence[str], vectors: MultiVectors) -> None: ...

    def seal(self) -> None: ...

    def search(
        self,
        query_vectors: MultiVectors,
        *,
        top_k: int,
    ) -> list[list[SearchHit]]: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class MultiVectorIndex(Protocol):
    """Factory for an isolated multi-vector index session."""

    def open(self, *, dimension: int) -> MultiVectorIndexSession: ...
