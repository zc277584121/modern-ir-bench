"""Sparse index contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from modern_ir_bench.embeddings.sparse import SparseVector
from modern_ir_bench.retrieval.types import SearchHit


class SparseIndexSession(Protocol):
    def add(self, ids: Sequence[str], vectors: Sequence[SparseVector]) -> None: ...

    def seal(self) -> None: ...

    def search(
        self,
        query_vectors: Sequence[SparseVector],
        *,
        top_k: int,
    ) -> list[list[SearchHit]]: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class SparseIndex(Protocol):
    def open(self) -> SparseIndexSession: ...
