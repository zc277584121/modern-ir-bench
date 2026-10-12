"""Rank-level fusion for independently prepared retrieval Solutions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.retrieval.types import RetrievalResource, SearchHit


@dataclass(frozen=True, kw_only=True)
class ReciprocalRankFusionSolution(Solution):
    """Fuse component rankings without assuming comparable score scales."""

    components: tuple[Solution, ...]
    weights: tuple[float, ...] | None = None
    candidate_depth: int = 50
    rank_constant: int = 60

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.components) < 2:
            raise ValueError("Rank fusion requires at least two components")
        weights = self.weights or tuple(1.0 for _ in self.components)
        if len(weights) != len(self.components) or any(weight <= 0 for weight in weights):
            raise ValueError("Rank fusion weights must be positive and match the components")
        if self.candidate_depth < 1 or self.rank_constant < 0:
            raise ValueError("candidate_depth must be positive and rank_constant must be non-negative")
        object.__setattr__(self, "weights", weights)

    def prepare(self, resources: Iterable[RetrievalResource[Any]]) -> ReciprocalRankFusionSession:
        rows = list(resources)
        sessions = []
        try:
            for component in self.components:
                sessions.append(component.prepare(rows))
        except Exception:
            for session in reversed(sessions):
                session.close()
            raise
        return ReciprocalRankFusionSession(
            sessions=tuple(sessions),
            weights=self.weights or (),
            candidate_depth=self.candidate_depth,
            rank_constant=self.rank_constant,
        )

    def release(self) -> None:
        for component in self.components:
            component.release()


class ReciprocalRankFusionSession:
    def __init__(
        self,
        *,
        sessions: tuple[Any, ...],
        weights: tuple[float, ...],
        candidate_depth: int,
        rank_constant: int,
    ) -> None:
        self.sessions = sessions
        self.weights = weights
        self.candidate_depth = candidate_depth
        self.rank_constant = rank_constant

    def search_batch(self, queries: Sequence[Any], *, top_k: int) -> list[list[SearchHit]]:
        depth = max(top_k, self.candidate_depth)
        component_results = [session.search_batch(queries, top_k=depth) for session in self.sessions]
        fused = []
        for query_index in range(len(queries)):
            scores: dict[str, float] = {}
            for weight, rankings in zip(self.weights, component_results, strict=True):
                for rank, hit in enumerate(rankings[query_index], start=1):
                    scores[hit.id] = scores.get(hit.id, 0.0) + weight / (self.rank_constant + rank)
            ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            fused.append([SearchHit(id=document_id, score=score) for document_id, score in ordered])
        return fused

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "reciprocal-rank-fusion",
            "weights": list(self.weights),
            "candidate_depth": self.candidate_depth,
            "rank_constant": self.rank_constant,
            "components": [dict(session.metadata) for session in self.sessions],
        }

    def close(self) -> None:
        with ExitStack() as stack:
            for session in self.sessions:
                stack.callback(session.close)
