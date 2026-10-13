"""Small model-facing contract for second-stage rerankers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class Reranker(Protocol):
    """Score candidate documents independently for each query."""

    id: str

    def score_batch(
        self,
        queries: Sequence[str],
        documents: Sequence[Sequence[str]],
    ) -> list[list[float]]: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...


def validate_batch(queries: Sequence[str], documents: Sequence[Sequence[str]]) -> None:
    if len(queries) != len(documents):
        raise ValueError("Queries and document groups must have equal lengths")
