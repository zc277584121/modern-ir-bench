from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from modern_ir_bench.core import Solution
from modern_ir_bench.retrieval import ReciprocalRankFusionSolution, RetrievalResource, SearchHit


class StaticSession:
    def __init__(self, rankings: list[list[str]]) -> None:
        self.rankings = rankings

    def search_batch(self, queries: Sequence[Any], *, top_k: int) -> list[list[SearchHit]]:
        assert len(queries) == len(self.rankings)
        return [
            [SearchHit(id=document_id, score=float(len(ids) - rank)) for rank, document_id in enumerate(ids[:top_k])]
            for ids in self.rankings
        ]

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"backend": "static"}

    def close(self) -> None:
        return None


@dataclass(frozen=True, kw_only=True)
class StaticSolution(Solution):
    rankings: tuple[tuple[str, ...], ...]

    def prepare(self, resources: Iterable[RetrievalResource[Any]]) -> StaticSession:
        assert list(resources)
        return StaticSession([list(ids) for ids in self.rankings])


def test_reciprocal_rank_fusion_combines_rankings_and_reuses_resources() -> None:
    solution = ReciprocalRankFusionSolution(
        id="rrf",
        title="RRF",
        components=(
            StaticSolution(id="first", title="First", rankings=(("a", "b", "c"),)),
            StaticSolution(id="second", title="Second", rankings=(("b", "d", "a"),)),
        ),
        candidate_depth=3,
        rank_constant=0,
    )
    resources = (RetrievalResource(id=str(index), value=index) for index in range(3))
    session = solution.prepare(resources)

    results = session.search_batch(["query"], top_k=3)

    assert [hit.id for hit in results[0]] == ["b", "a", "d"]
    assert session.metadata["backend"] == "reciprocal-rank-fusion"
    session.close()
