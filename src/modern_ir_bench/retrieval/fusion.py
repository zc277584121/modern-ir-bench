"""Rank-level fusion for independently prepared retrieval Solutions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.retrieval.candidate_cache import SqliteCandidateCache
from modern_ir_bench.retrieval.types import RetrievalResource, SearchHit


@dataclass(frozen=True, kw_only=True)
class ReciprocalRankFusionSolution(Solution):
    """Fuse component rankings without assuming comparable score scales."""

    components: tuple[Solution, ...]
    weights: tuple[float, ...] | None = None
    candidate_depth: int = 50
    rank_constant: int = 60
    candidate_cache_path: Path | None = None
    candidate_cache_namespace: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_components(self.components)
        weights = self.weights or tuple(1.0 for _ in self.components)
        if len(weights) != len(self.components) or any(weight <= 0 for weight in weights):
            raise ValueError("Rank fusion weights must be positive and match the components")
        if self.candidate_depth < 1 or self.rank_constant < 0:
            raise ValueError("candidate_depth must be positive and rank_constant must be non-negative")
        if (self.candidate_cache_path is None) != (self.candidate_cache_namespace is None):
            raise ValueError("Candidate cache path and namespace must be configured together")
        object.__setattr__(self, "weights", weights)

    def prepare(self, resources: Iterable[RetrievalResource[Any]]) -> ReciprocalRankFusionSession:
        rows, sessions, cache = _prepare_components(
            components=self.components,
            resources=resources,
            cache_path=self.candidate_cache_path,
            cache_namespace=self.candidate_cache_namespace,
        )
        return ReciprocalRankFusionSession(
            components=self.components,
            resources=rows,
            sessions=sessions,
            weights=self.weights or (),
            candidate_depth=self.candidate_depth,
            rank_constant=self.rank_constant,
            candidate_cache=cache,
        )

    def release(self) -> None:
        for component in self.components:
            component.release()


@dataclass(frozen=True, kw_only=True)
class WeightedScoreFusionSolution(Solution):
    """Fuse normalized component scores with explicit branch weights."""

    components: tuple[Solution, ...]
    weights: tuple[float, ...]
    score_directions: tuple[Literal["higher", "lower"], ...] | None = None
    candidate_depth: int = 50
    candidate_cache_path: Path | None = None
    candidate_cache_namespace: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_components(self.components)
        if len(self.weights) != len(self.components) or any(weight < 0 for weight in self.weights):
            raise ValueError("Score fusion weights must be non-negative and match the components")
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("At least one score fusion weight must be positive")
        if self.score_directions is not None and (
            len(self.score_directions) != len(self.components)
            or any(direction not in {"higher", "lower"} for direction in self.score_directions)
        ):
            raise ValueError("Score directions must be 'higher' or 'lower' and match the components")
        if self.candidate_depth < 1:
            raise ValueError("candidate_depth must be positive")
        if (self.candidate_cache_path is None) != (self.candidate_cache_namespace is None):
            raise ValueError("Candidate cache path and namespace must be configured together")
        object.__setattr__(self, "weights", tuple(weight / total for weight in self.weights))
        object.__setattr__(
            self,
            "score_directions",
            self.score_directions or tuple("higher" for _ in self.components),
        )

    def prepare(self, resources: Iterable[RetrievalResource[Any]]) -> WeightedScoreFusionSession:
        rows, sessions, cache = _prepare_components(
            components=self.components,
            resources=resources,
            cache_path=self.candidate_cache_path,
            cache_namespace=self.candidate_cache_namespace,
        )
        return WeightedScoreFusionSession(
            components=self.components,
            resources=rows,
            sessions=sessions,
            weights=self.weights,
            score_directions=self.score_directions or (),
            candidate_depth=self.candidate_depth,
            candidate_cache=cache,
        )

    def release(self) -> None:
        for component in self.components:
            component.release()


def _validate_components(components: tuple[Solution, ...]) -> None:
    if len(components) < 2:
        raise ValueError("Fusion requires at least two components")
    component_ids = [component.id for component in components]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("Fusion component ids must be unique")


def _prepare_components(
    *,
    components: tuple[Solution, ...],
    resources: Iterable[RetrievalResource[Any]],
    cache_path: Path | None,
    cache_namespace: str | None,
) -> tuple[list[RetrievalResource[Any]], list[Any | None], SqliteCandidateCache | None]:
    rows = list(resources)
    cache = (
        SqliteCandidateCache(cache_path, namespace=cache_namespace)
        if cache_path is not None and cache_namespace is not None
        else None
    )
    sessions: list[Any | None] = [None] * len(components)
    if cache is None:
        try:
            for index, component in enumerate(components):
                sessions[index] = component.prepare(rows)
        except Exception:
            for session in reversed(sessions):
                if session is not None:
                    session.close()
            raise
    return rows, sessions, cache


class FusionSession:
    def __init__(
        self,
        *,
        components: tuple[Solution, ...],
        resources: list[RetrievalResource[Any]],
        sessions: list[Any | None],
        candidate_depth: int,
        candidate_cache: SqliteCandidateCache | None,
    ) -> None:
        self.components = components
        self.resources = resources
        self.corpus_key = SqliteCandidateCache.corpus_key(resources)
        self.sessions = sessions
        self.candidate_depth = candidate_depth
        self.candidate_cache = candidate_cache
        self.cache_hits = 0
        self.cache_misses = 0

    def _component_results(
        self,
        index: int,
        component: Solution,
        queries: Sequence[Any],
        depth: int,
    ) -> list[list[SearchHit]]:
        cached = (
            self.candidate_cache.read(
                corpus_key=self.corpus_key,
                component_id=component.id,
                queries=queries,
                depth=depth,
            )
            if self.candidate_cache is not None
            else {}
        )
        self.cache_hits += len(cached)
        missing_indexes = [query_index for query_index in range(len(queries)) if query_index not in cached]
        self.cache_misses += len(missing_indexes)
        if missing_indexes:
            session = self._session(index)
            missing_queries = [queries[query_index] for query_index in missing_indexes]
            computed = session.search_batch(missing_queries, top_k=depth)
            if len(computed) != len(missing_queries):
                raise RuntimeError(f"Fusion component {component.id} returned the wrong number of rankings")
            if self.candidate_cache is not None:
                self.candidate_cache.write(
                    corpus_key=self.corpus_key,
                    component_id=component.id,
                    queries=missing_queries,
                    depth=depth,
                    rankings=computed,
                )
            cached.update(zip(missing_indexes, computed, strict=True))
        return [cached[query_index] for query_index in range(len(queries))]

    def _session(self, index: int) -> Any:
        session = self.sessions[index]
        if session is None:
            session = self.components[index].prepare(self.resources)
            self.sessions[index] = session
        return session

    def _metadata(self) -> dict[str, Any]:
        return {
            "candidate_depth": self.candidate_depth,
            "components": [
                dict(session.metadata) if session is not None else {"solution_id": component.id, "cached": True}
                for component, session in zip(self.components, self.sessions, strict=True)
            ],
            "candidate_cache": {
                "enabled": self.candidate_cache is not None,
                "namespace": self.candidate_cache.namespace if self.candidate_cache is not None else None,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
            },
        }

    def close(self) -> None:
        with ExitStack() as stack:
            for session in self.sessions:
                if session is not None:
                    stack.callback(session.close)


class ReciprocalRankFusionSession(FusionSession):
    def __init__(
        self,
        *,
        components: tuple[Solution, ...],
        resources: list[RetrievalResource[Any]],
        sessions: list[Any | None],
        weights: tuple[float, ...],
        candidate_depth: int,
        rank_constant: int,
        candidate_cache: SqliteCandidateCache | None,
    ) -> None:
        super().__init__(
            components=components,
            resources=resources,
            sessions=sessions,
            candidate_depth=candidate_depth,
            candidate_cache=candidate_cache,
        )
        self.weights = weights
        self.rank_constant = rank_constant

    def search_batch(self, queries: Sequence[Any], *, top_k: int) -> list[list[SearchHit]]:
        depth = max(top_k, self.candidate_depth)
        component_results = [
            self._component_results(index, component, queries, depth) for index, component in enumerate(self.components)
        ]
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
            **self._metadata(),
            "backend": "reciprocal-rank-fusion",
            "weights": list(self.weights),
            "rank_constant": self.rank_constant,
        }


class WeightedScoreFusionSession(FusionSession):
    def __init__(
        self,
        *,
        components: tuple[Solution, ...],
        resources: list[RetrievalResource[Any]],
        sessions: list[Any | None],
        weights: tuple[float, ...],
        score_directions: tuple[Literal["higher", "lower"], ...],
        candidate_depth: int,
        candidate_cache: SqliteCandidateCache | None,
    ) -> None:
        super().__init__(
            components=components,
            resources=resources,
            sessions=sessions,
            candidate_depth=candidate_depth,
            candidate_cache=candidate_cache,
        )
        self.weights = weights
        self.score_directions = score_directions

    def search_batch(self, queries: Sequence[Any], *, top_k: int) -> list[list[SearchHit]]:
        depth = max(top_k, self.candidate_depth)
        component_results = [
            self._component_results(index, component, queries, depth) for index, component in enumerate(self.components)
        ]
        fused = []
        for query_index in range(len(queries)):
            scores: dict[str, float] = {}
            for weight, direction, rankings in zip(
                self.weights,
                self.score_directions,
                component_results,
                strict=True,
            ):
                normalized = _min_max_scores(rankings[query_index], direction=direction)
                for document_id, score in normalized.items():
                    scores[document_id] = scores.get(document_id, 0.0) + weight * score
            ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            fused.append([SearchHit(id=document_id, score=score) for document_id, score in ordered])
        return fused

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            **self._metadata(),
            "backend": "weighted-score-fusion",
            "weights": list(self.weights),
            "normalization": "per-query-min-max",
            "score_directions": list(self.score_directions),
        }


def _min_max_scores(
    rankings: Sequence[SearchHit],
    *,
    direction: Literal["higher", "lower"],
) -> dict[str, float]:
    if not rankings:
        return {}
    values = [hit.score for hit in rankings]
    low = min(values)
    high = max(values)
    if high == low:
        return {hit.id: 0.0 for hit in rankings}
    if direction == "higher":
        return {hit.id: (hit.score - low) / (high - low) for hit in rankings}
    return {hit.id: (high - hit.score) / (high - low) for hit in rankings}
