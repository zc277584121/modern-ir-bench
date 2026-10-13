"""Second-stage document reranking composed with any first-stage Solution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.rerankers import Reranker
from modern_ir_bench.rerankers.cache import SqliteRerankCache
from modern_ir_bench.retrieval.candidate_cache import SqliteCandidateCache
from modern_ir_bench.retrieval.types import RetrievalResource, SearchHit


@dataclass(frozen=True, kw_only=True)
class RerankingSolution(Solution):
    """Retrieve a bounded candidate set, then score complete candidate documents."""

    candidate_solution: Solution
    reranker: Reranker
    document_input: Callable[[Any], str]
    query_input: Callable[[Any], str]
    candidate_depth: int = 50
    candidate_cache_path: Path | None = None
    candidate_cache_namespace: str | None = None
    rerank_cache_path: Path | None = None
    rerank_cache_namespace: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.candidate_depth < 1:
            raise ValueError("Candidate depth must be positive")
        if (self.candidate_cache_path is None) != (self.candidate_cache_namespace is None):
            raise ValueError("Candidate cache path and namespace must be configured together")
        if (self.rerank_cache_path is None) != (self.rerank_cache_namespace is None):
            raise ValueError("Rerank cache path and namespace must be configured together")

    def prepare(self, resources: Iterable[RetrievalResource[Any]]) -> RerankingSession:
        rows = list(resources)
        if len({row.id for row in rows}) != len(rows):
            raise ValueError("Reranking resources must have unique ids")
        return RerankingSession(
            candidate_solution=self.candidate_solution,
            reranker=self.reranker,
            resources=rows,
            document_input=self.document_input,
            query_input=self.query_input,
            candidate_depth=self.candidate_depth,
            candidate_cache=(
                SqliteCandidateCache(self.candidate_cache_path, namespace=self.candidate_cache_namespace)
                if self.candidate_cache_path is not None and self.candidate_cache_namespace is not None
                else None
            ),
            rerank_cache=(
                SqliteRerankCache(self.rerank_cache_path, namespace=self.rerank_cache_namespace)
                if self.rerank_cache_path is not None and self.rerank_cache_namespace is not None
                else None
            ),
        )

    def release(self) -> None:
        self.candidate_solution.release()


class RerankingSession:
    def __init__(
        self,
        *,
        candidate_solution: Solution,
        reranker: Reranker,
        resources: list[RetrievalResource[Any]],
        document_input: Callable[[Any], str],
        query_input: Callable[[Any], str],
        candidate_depth: int,
        candidate_cache: SqliteCandidateCache | None,
        rerank_cache: SqliteRerankCache | None,
    ) -> None:
        self.candidate_solution = candidate_solution
        self.reranker = reranker
        self.resources = resources
        self.documents = {row.id: document_input(row.value) for row in resources}
        self.query_input = query_input
        self.candidate_depth = candidate_depth
        self.candidate_cache = candidate_cache
        self.rerank_cache = rerank_cache
        self.corpus_key = SqliteCandidateCache.corpus_key(resources)
        self.candidate_session: Any | None = None
        self.candidate_cache_hits = 0
        self.candidate_cache_misses = 0
        self.rerank_cache_hits = 0
        self.rerank_cache_misses = 0

    def search_batch(self, queries: Sequence[Any], *, top_k: int) -> list[list[SearchHit]]:
        depth = max(top_k, self.candidate_depth)
        candidates = self._candidates(queries, depth)
        query_texts = [self.query_input(query) for query in queries]
        score_rows: list[dict[str, float]] = []
        missing_ids: list[list[str]] = []
        missing_texts: list[list[str]] = []
        for query, ranking in zip(query_texts, candidates, strict=True):
            documents = [(hit.id, self.documents[hit.id]) for hit in ranking]
            cached = (
                self.rerank_cache.read(reranker_id=self.reranker.id, query=query, documents=documents)
                if self.rerank_cache is not None
                else {}
            )
            missing = [(document_id, text) for document_id, text in documents if document_id not in cached]
            self.rerank_cache_hits += len(cached)
            self.rerank_cache_misses += len(missing)
            score_rows.append(cached)
            missing_ids.append([document_id for document_id, _ in missing])
            missing_texts.append([text for _, text in missing])

        computed = self.reranker.score_batch(query_texts, missing_texts)
        if len(computed) != len(query_texts):
            raise RuntimeError(f"Reranker {self.reranker.id} returned the wrong number of query rows")
        for query, ids, texts, scores, row in zip(
            query_texts,
            missing_ids,
            missing_texts,
            computed,
            score_rows,
            strict=True,
        ):
            if len(scores) != len(ids):
                raise RuntimeError(f"Reranker {self.reranker.id} returned incomplete document scores")
            values = dict(zip(ids, map(float, scores), strict=True))
            row.update(values)
            if self.rerank_cache is not None and ids:
                self.rerank_cache.write(
                    reranker_id=self.reranker.id,
                    query=query,
                    documents=list(zip(ids, texts, strict=True)),
                    scores=values,
                )

        reranked = []
        for ranking, scores in zip(candidates, score_rows, strict=True):
            original_rank = {hit.id: rank for rank, hit in enumerate(ranking)}
            ordered = sorted(
                scores.items(),
                key=lambda item: (-item[1], original_rank[item[0]], item[0]),
            )[:top_k]
            reranked.append([SearchHit(id=document_id, score=score) for document_id, score in ordered])
        return reranked

    def _candidates(self, queries: Sequence[Any], depth: int) -> list[list[SearchHit]]:
        cached = (
            self.candidate_cache.read(
                corpus_key=self.corpus_key,
                component_id=self.candidate_solution.id,
                queries=queries,
                depth=depth,
            )
            if self.candidate_cache is not None
            else {}
        )
        self.candidate_cache_hits += len(cached)
        missing_indexes = [index for index in range(len(queries)) if index not in cached]
        self.candidate_cache_misses += len(missing_indexes)
        if missing_indexes:
            if self.candidate_session is None:
                self.candidate_session = self.candidate_solution.prepare(self.resources)
            missing_queries = [queries[index] for index in missing_indexes]
            rankings = self.candidate_session.search_batch(missing_queries, top_k=depth)
            if len(rankings) != len(missing_queries):
                raise RuntimeError("Candidate Solution returned the wrong number of rankings")
            if self.candidate_cache is not None:
                self.candidate_cache.write(
                    corpus_key=self.corpus_key,
                    component_id=self.candidate_solution.id,
                    queries=missing_queries,
                    depth=depth,
                    rankings=rankings,
                )
            cached.update(zip(missing_indexes, rankings, strict=True))
        return [cached[index] for index in range(len(queries))]

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "retrieve-then-rerank",
            "candidate_depth": self.candidate_depth,
            "candidate_solution": (
                dict(self.candidate_session.metadata)
                if self.candidate_session is not None
                else {"solution_id": self.candidate_solution.id, "cached": True}
            ),
            "reranker": dict(self.reranker.metadata),
            "candidate_cache": {
                "enabled": self.candidate_cache is not None,
                "namespace": self.candidate_cache.namespace if self.candidate_cache is not None else None,
                "hits": self.candidate_cache_hits,
                "misses": self.candidate_cache_misses,
            },
            "rerank_cache": {
                "enabled": self.rerank_cache is not None,
                "namespace": self.rerank_cache.namespace if self.rerank_cache is not None else None,
                "hits": self.rerank_cache_hits,
                "misses": self.rerank_cache_misses,
            },
        }

    def close(self) -> None:
        if self.candidate_session is not None:
            self.candidate_session.close()
