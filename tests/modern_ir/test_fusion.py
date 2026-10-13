from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True, kw_only=True)
class NeverPrepareSolution(Solution):
    def prepare(self, resources: Iterable[RetrievalResource[Any]]) -> StaticSession:
        raise AssertionError("cached candidates should avoid preparing retrieval components")


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


def _cached_solution(
    cache_path: Path,
    *,
    weights: tuple[float, float] = (1.0, 1.0),
    namespace: str = "test-corpus:components-v1",
    components: tuple[Solution, Solution] | None = None,
    candidate_depth: int = 3,
) -> ReciprocalRankFusionSolution:
    return ReciprocalRankFusionSolution(
        id="rrf",
        title="RRF",
        components=components
        or (
            StaticSolution(id="first", title="First", rankings=(("a", "b", "c", "e"),)),
            StaticSolution(id="second", title="Second", rankings=(("b", "d", "a", "f"),)),
        ),
        weights=weights,
        candidate_depth=candidate_depth,
        rank_constant=0,
        candidate_cache_path=cache_path,
        candidate_cache_namespace=namespace,
    )


def test_fusion_reuses_cached_component_candidates_when_weights_change(tmp_path: Path) -> None:
    cache_path = tmp_path / "candidates.sqlite3"
    first_session = _cached_solution(cache_path).prepare([RetrievalResource(id="doc", value="content")])
    first_results = first_session.search_batch(["query"], top_k=3)
    first_session.close()

    cached_components = (
        NeverPrepareSolution(id="first", title="First"),
        NeverPrepareSolution(id="second", title="Second"),
    )
    second_session = _cached_solution(
        cache_path,
        weights=(10.0, 1.0),
        components=cached_components,
    ).prepare([RetrievalResource(id="doc", value="content")])
    second_results = second_session.search_batch(["query"], top_k=3)

    assert [hit.id for hit in first_results[0]] == ["b", "a", "d"]
    assert [hit.id for hit in second_results[0]] == ["a", "b", "c"]
    assert second_session.metadata["candidate_cache"] == {
        "enabled": True,
        "namespace": "test-corpus:components-v1",
        "hits": 2,
        "misses": 0,
    }
    second_session.close()


def test_fusion_cache_isolated_by_namespace_and_candidate_depth(tmp_path: Path) -> None:
    cache_path = tmp_path / "candidates.sqlite3"
    initial = _cached_solution(cache_path).prepare([RetrievalResource(id="doc", value="content")])
    initial.search_batch(["query"], top_k=2)
    initial.close()

    deeper = _cached_solution(cache_path, candidate_depth=4).prepare([RetrievalResource(id="doc", value="content")])
    deeper.search_batch(["query"], top_k=2)
    assert deeper.metadata["candidate_cache"] == {
        "enabled": True,
        "namespace": "test-corpus:components-v1",
        "hits": 0,
        "misses": 2,
    }
    deeper.close()

    different_namespace = _cached_solution(cache_path, namespace="other-corpus:components-v1").prepare(
        [RetrievalResource(id="doc", value="content")]
    )
    different_namespace.search_batch(["query"], top_k=2)
    assert different_namespace.metadata["candidate_cache"] == {
        "enabled": True,
        "namespace": "other-corpus:components-v1",
        "hits": 0,
        "misses": 2,
    }
    different_namespace.close()
