from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modern_ir_bench.core import Solution
from modern_ir_bench.retrieval import RerankingSolution, RetrievalResource, SearchHit


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
        return StaticSession([list(row) for row in self.rankings])


@dataclass(frozen=True, kw_only=True)
class NeverPrepareSolution(Solution):
    def prepare(self, resources: Iterable[RetrievalResource[Any]]) -> StaticSession:
        raise AssertionError("cached candidates should avoid preparing the first stage")


class TermReranker:
    id = "term-reranker:v1"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.scored_pairs = 0

    def score_batch(self, queries: Sequence[str], documents: Sequence[Sequence[str]]) -> list[list[float]]:
        if self.fail and any(documents):
            raise AssertionError("cached rerank scores should avoid model inference")
        self.scored_pairs += sum(map(len, documents))
        return [[float(document.count(query)) for document in rows] for query, rows in zip(queries, documents)]

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"backend": "term"}


def _resources() -> list[RetrievalResource[str]]:
    return [
        RetrievalResource(id="a", value="plain text"),
        RetrievalResource(id="b", value="target target"),
        RetrievalResource(id="c", value="one target"),
    ]


def _solution(
    tmp_path: Path,
    *,
    candidate: Solution,
    reranker: TermReranker,
) -> RerankingSolution:
    return RerankingSolution(
        id="reranked",
        title="Reranked",
        candidate_solution=candidate,
        reranker=reranker,
        document_input=str,
        query_input=str,
        candidate_depth=3,
        candidate_cache_path=tmp_path / "candidates.sqlite3",
        candidate_cache_namespace="test-candidates-v1",
        rerank_cache_path=tmp_path / "rerank.sqlite3",
        rerank_cache_namespace="test-rerank-v1",
    )


def test_reranking_reorders_complete_candidate_documents(tmp_path: Path) -> None:
    reranker = TermReranker()
    session = _solution(
        tmp_path,
        candidate=StaticSolution(id="first-stage", title="First", rankings=(("a", "c", "b"),)),
        reranker=reranker,
    ).prepare(_resources())

    results = session.search_batch(["target"], top_k=2)

    assert [hit.id for hit in results[0]] == ["b", "c"]
    assert reranker.scored_pairs == 3
    assert session.metadata["candidate_cache"]["misses"] == 1
    assert session.metadata["rerank_cache"]["misses"] == 3
    session.close()


def test_reranking_reuses_candidate_and_pair_scores(tmp_path: Path) -> None:
    first = _solution(
        tmp_path,
        candidate=StaticSolution(id="first-stage", title="First", rankings=(("a", "c", "b"),)),
        reranker=TermReranker(),
    ).prepare(_resources())
    first.search_batch(["target"], top_k=2)
    first.close()

    second = _solution(
        tmp_path,
        candidate=NeverPrepareSolution(id="first-stage", title="First"),
        reranker=TermReranker(fail=True),
    ).prepare(_resources())
    results = second.search_batch(["target"], top_k=2)

    assert [hit.id for hit in results[0]] == ["b", "c"]
    assert second.metadata["candidate_cache"]["hits"] == 1
    assert second.metadata["rerank_cache"]["hits"] == 3
    second.close()


def test_rerank_cache_invalidates_changed_document_content(tmp_path: Path) -> None:
    initial = _solution(
        tmp_path,
        candidate=StaticSolution(id="first-stage", title="First", rankings=(("a", "c", "b"),)),
        reranker=TermReranker(),
    ).prepare(_resources())
    initial.search_batch(["target"], top_k=2)
    initial.close()

    changed = _resources()
    changed[1] = RetrievalResource(id="b", value="no longer relevant")
    reranker = TermReranker()
    session = _solution(
        tmp_path,
        candidate=StaticSolution(id="first-stage", title="First", rankings=(("a", "c", "b"),)),
        reranker=reranker,
    ).prepare(changed)
    session.search_batch(["target"], top_k=2)

    assert reranker.scored_pairs == 1
    assert session.metadata["rerank_cache"]["hits"] == 2
    assert session.metadata["rerank_cache"]["misses"] == 1
    session.close()
