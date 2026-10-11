"""Derive leaderboard slices from retrieval evidence.

This module contains presentation analysis only. It never performs retrieval and it
never changes Qrels; every slice is recomputed from an existing Top-K ranking.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

METRIC_ORDER = ("ndcg@10", "recall@1", "recall@5", "recall@10", "mrr@10")
METRIC_LABELS = {
    "ndcg@10": "NDCG@10",
    "recall@1": "Recall@1",
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "mrr@10": "MRR@10",
}


@dataclass(frozen=True)
class Dimension:
    """One labeled field exposed as a leaderboard slice."""

    id: str
    label: str
    values: tuple[tuple[str, str], ...] = ()


def _title(value: str) -> str:
    abbreviations = {"ai": "AI", "hr": "HR", "qa": "Q&A"}
    return " ".join(
        abbreviations.get(word, word.capitalize()) for word in value.replace("/", " / ").replace("_", " ").split()
    )


def _query_metrics(expected: set[str], ranked: Sequence[str]) -> dict[str, float]:
    if not expected:
        raise ValueError("Each query must have at least one relevant document")
    recalls = {f"recall@{cutoff}": len(expected.intersection(ranked[:cutoff])) / len(expected) for cutoff in (1, 5, 10)}
    gains = [float(document_id in expected) for document_id in ranked[:10]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(expected), 10) + 1))
    return {
        "ndcg@10": dcg / ideal,
        **recalls,
        "mrr@10": next(
            (1 / rank for rank, document_id in enumerate(ranked[:10], start=1) if document_id in expected),
            0.0,
        ),
    }


def _dimension_values(dimension: Dimension, rows: Mapping[str, Mapping[str, str]]) -> tuple[tuple[str, str], ...]:
    if dimension.values:
        return dimension.values
    return tuple((value, _title(value)) for value in sorted({row[dimension.id] for row in rows.values()}))


def _metric_rows(
    metrics_by_solution: Mapping[str, Mapping[str, Mapping[str, float]]],
    query_ids: set[str],
) -> list[dict[str, object]]:
    if not query_ids:
        raise ValueError("A published query slice must not be empty")
    return [
        {
            "solution_id": solution_id,
            "metric_id": metric_id,
            "metric_label": METRIC_LABELS[metric_id],
            "value": round(statistics.mean(per_query[query_id][metric_id] for query_id in query_ids), 8),
            "primary": metric_id == "ndcg@10",
        }
        for solution_id, per_query in metrics_by_solution.items()
        for metric_id in METRIC_ORDER
    ]


def build_retrieval_breakdowns(
    *,
    task_id: str,
    dataset_id: str,
    dataset_version: str,
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    relevant_by_query: Mapping[str, set[str]],
    query_labels: Mapping[str, Mapping[str, str]],
    document_labels: Mapping[str, Mapping[str, str]],
    query_dimensions: Sequence[Dimension] = (),
    document_dimensions: Sequence[Dimension] = (),
) -> list[dict[str, Any]]:
    """Compute selected query and relevant-document slices from saved rankings."""

    query_ids = set(query_labels)
    if query_ids != set(relevant_by_query):
        raise ValueError("Query labels and relevance labels must cover the same query ids")
    for solution_id, solution_rankings in rankings.items():
        if set(solution_rankings) != query_ids:
            raise ValueError(f"Rankings for {solution_id} do not cover every query")

    metrics_by_solution = {
        solution_id: {
            query_id: _query_metrics(relevant_by_query[query_id], ranked_ids)
            for query_id, ranked_ids in solution_rankings.items()
        }
        for solution_id, solution_rankings in rankings.items()
    }
    context = {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
    }
    breakdowns: list[dict[str, Any]] = []

    for dimension in query_dimensions:
        values = []
        for value_id, value_label in _dimension_values(dimension, query_labels):
            selected = {query_id for query_id, labels in query_labels.items() if labels[dimension.id] == value_id}
            values.append(
                {
                    "id": value_id,
                    "label": value_label,
                    "count": len(selected),
                    "results": _metric_rows(metrics_by_solution, selected),
                }
            )
        breakdowns.append(
            {
                "id": dimension.id,
                "label": dimension.label,
                "category": "query",
                "description": f"Scores for queries grouped by {dimension.label.lower()}.",
                "primary_metric": "ndcg@10",
                **context,
                "values": values,
            }
        )

    for dimension in document_dimensions:
        values = []
        for value_id, value_label in _dimension_values(dimension, document_labels):
            targets = {
                query_id: {
                    document_id
                    for document_id in relevant_ids
                    if document_labels[document_id][dimension.id] == value_id
                }
                for query_id, relevant_ids in relevant_by_query.items()
            }
            targets = {query_id: document_ids for query_id, document_ids in targets.items() if document_ids}
            if not targets:
                continue
            result_rows = []
            for solution_id, solution_rankings in rankings.items():
                for cutoff in (1, 5, 10):
                    scores = [
                        len(set(solution_rankings[query_id][:cutoff]).intersection(target_ids)) / len(target_ids)
                        for query_id, target_ids in targets.items()
                    ]
                    result_rows.append(
                        {
                            "solution_id": solution_id,
                            "metric_id": f"target_recall@{cutoff}",
                            "metric_label": f"Target Recall@{cutoff}",
                            "value": round(statistics.mean(scores), 8),
                            "primary": cutoff == 10,
                        }
                    )
            values.append(
                {
                    "id": value_id,
                    "label": value_label,
                    "count": len(targets),
                    "target_count": sum(len(document_ids) for document_ids in targets.values()),
                    "results": result_rows,
                }
            )
        breakdowns.append(
            {
                "id": f"relevant_document_{dimension.id}",
                "label": dimension.label,
                "category": "relevant_documents",
                "description": (
                    "Recall of relevant documents in this group within each saved Top-10; retrieval is not rerun."
                ),
                "primary_metric": "target_recall@10",
                **context,
                "values": values,
            }
        )
    return breakdowns
