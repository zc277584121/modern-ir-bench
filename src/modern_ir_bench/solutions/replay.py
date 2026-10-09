"""Replay immutable saved rankings as benchmark Solutions."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.retrieval.types import RetrievalResource, SearchHit


class SavedRankingSession:
    def __init__(self, rankings: Mapping[str, Sequence[str]]) -> None:
        self.rankings = rankings

    def search_batch(self, queries: Sequence[str], *, top_k: int) -> list[list[SearchHit]]:
        missing = [query for query in queries if query not in self.rankings]
        if missing:
            raise ValueError(f"saved rankings do not cover queries: {missing[:3]}")
        return [
            [
                SearchHit(id=doc_id, score=1.0 / rank)
                for rank, doc_id in enumerate(self.rankings[query][:top_k], start=1)
            ]
            for query in queries
        ]

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"backend": "saved-ranking-replay"}

    def close(self) -> None:
        return None


@dataclass(frozen=True, kw_only=True)
class SavedRankingSolution(Solution):
    """A reproducibility Solution backed by saved query-to-document rankings."""

    rankings: Mapping[str, Sequence[str]]

    def prepare(self, resources: Iterable[RetrievalResource[str]]) -> SavedRankingSession:
        resource_ids = {resource.id for resource in resources}
        ranked_ids = {doc_id for ranking in self.rankings.values() for doc_id in ranking}
        if not ranked_ids.issubset(resource_ids):
            unknown = sorted(ranked_ids - resource_ids)
            raise ValueError(f"saved rankings contain unknown documents: {unknown[:10]}")
        return SavedRankingSession(self.rankings)


def load_release_replay_solutions(release_dir: Path) -> list[SavedRankingSolution]:
    """Build one replay Solution per saved baseline route in an immutable release."""
    release_dir = release_dir.resolve()
    baseline = json.loads((release_dir / "baseline-results.json").read_text(encoding="utf-8"))
    return build_saved_ranking_solutions(
        queries=_read_jsonl(release_dir / "queries.jsonl"),
        rankings=_read_jsonl(release_dir / "baseline-observations.jsonl"),
        solution_metadata=baseline["solutions"],
    )


def build_saved_ranking_solutions(
    *,
    queries: Iterable[Mapping[str, Any]],
    rankings: Iterable[Mapping[str, Any]],
    solution_metadata: Mapping[str, Mapping[str, Any]],
) -> list[SavedRankingSolution]:
    """Build replay Solutions from public query and ranking rows."""
    query_text = {str(row["query_id"]): str(row["text"]) for row in queries}
    if len(query_text) != len(set(query_text.values())):
        raise ValueError("saved ranking replay requires unique query text")

    by_solution: dict[str, dict[str, list[str]]] = {}
    for row in rankings:
        solution_id = str(row["solution_id"])
        by_solution.setdefault(solution_id, {})[query_text[str(row["query_id"])]] = list(row["ranked_doc_ids"])
    expected_queries = set(query_text.values())
    for solution_id in solution_metadata:
        actual_queries = set(by_solution.get(solution_id, {}))
        if actual_queries != expected_queries:
            raise ValueError(
                f"saved rankings for {solution_id} are incomplete: "
                f"missing={sorted(expected_queries - actual_queries)[:3]}"
            )
    return [
        SavedRankingSolution(
            id=solution_id,
            title=str(metadata["title"]),
            description=f"Replay of the immutable {metadata['route']} ranking.",
            rankings=by_solution[solution_id],
        )
        for solution_id, metadata in solution_metadata.items()
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
