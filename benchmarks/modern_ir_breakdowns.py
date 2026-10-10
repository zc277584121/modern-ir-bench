"""Leaderboard breakdowns derived from immutable rankings and Dataset labels."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from modern_ir_bench.core.result import RunReport
from modern_ir_bench.results import RankingArtifact

QUERY_BREAKDOWNS = (
    ("language", "Language", (("zh", "Chinese"), ("en", "English"))),
    (
        "query_intent",
        "Intent",
        (
            ("understand", "Understand"),
            ("act", "Act"),
            ("lookup", "Lookup"),
            ("decide", "Decide"),
            ("verify", "Verify"),
            ("synthesize", "Synthesize"),
        ),
    ),
    (
        "expression",
        "Expression",
        (
            ("natural_question", "Natural question"),
            ("search_phrase", "Search phrase"),
            ("contextual_request", "Contextual request"),
        ),
    ),
    (
        "constraint_level",
        "Constraint level",
        (("single", "Single"), ("compound", "Compound"), ("light", "Light")),
    ),
)
RELEVANT_DOCUMENT_BREAKDOWNS = (
    ("language", "Language"),
    ("domain", "Domain"),
    ("form", "Form"),
    ("length", "Length"),
)


def _query_metrics(expected: set[str], ranked: Iterable[str], k: int = 10) -> dict[str, float]:
    ranking = list(ranked)
    recalls = {
        f"recall@{cutoff}": len(expected.intersection(ranking[:cutoff])) / len(expected) for cutoff in (1, 5, 10)
    }
    gains = [float(document_id in expected) for document_id in ranking[:k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(expected), k) + 1))
    return {
        "ndcg@10": dcg / ideal,
        **recalls,
        "mrr@10": next(
            (1 / rank for rank, document_id in enumerate(ranking[:k], start=1) if document_id in expected),
            0.0,
        ),
    }


def _target_recall(ranked: Iterable[str], targets: set[str], cutoff: int) -> float:
    return len(set(list(ranked)[:cutoff]).intersection(targets)) / len(targets)


def _label(value: str) -> str:
    abbreviations = {"ai": "AI", "hr": "HR", "qa": "Q&A"}
    return " ".join(
        abbreviations.get(word, word.capitalize()) for word in value.replace("/", " / ").replace("_", " ").split()
    )


def _context(report: RunReport) -> dict[str, Any]:
    record = report.records[0]
    return {
        "task_id": record["task_id"],
        "dataset_id": record["dataset_id"],
        "dataset_version": record["dataset_version"],
    }


def build_breakdowns(
    *,
    documents: Iterable[Mapping[str, Any]],
    queries: Iterable[Mapping[str, Any]],
    qrels: Iterable[Mapping[str, Any]],
    artifact: RankingArtifact,
    report: RunReport,
) -> list[dict[str, Any]]:
    document_rows = list(documents)
    query_rows = list(queries)
    qrel_rows = list(qrels)
    query_by_id = {str(row["query_id"]): row for row in query_rows}
    dimensions_by_document = {str(row["doc_id"]): row["dimensions"] for row in document_rows}
    relevant_by_query: dict[str, set[str]] = defaultdict(set)
    for row in qrel_rows:
        if int(row["relevance"]) > 0:
            relevant_by_query[str(row["query_id"])].add(str(row["doc_id"]))
    per_query = {
        solution_id: {
            query_id: _query_metrics(relevant_by_query[query_id], ranked_ids)
            for query_id, ranked_ids in rankings.items()
        }
        for solution_id, rankings in artifact.rankings.items()
    }
    metric_labels = {str(row["metric_id"]): str(row["metric_label"]) for row in report.records}
    metric_order = ["ndcg@10", "recall@1", "recall@5", "recall@10", "mrr@10"]
    context = _context(report)
    breakdowns: list[dict[str, Any]] = []

    for field, label, values in QUERY_BREAKDOWNS:
        groups = []
        for value_id, value_label in values:
            query_ids = {query_id for query_id, row in query_by_id.items() if str(row[field]) == value_id}
            results = []
            for solution_id, metrics_by_query in per_query.items():
                for metric_id in metric_order:
                    results.append(
                        {
                            "solution_id": solution_id,
                            "metric_id": metric_id,
                            "metric_label": metric_labels[metric_id],
                            "value": round(
                                statistics.mean(metrics_by_query[query_id][metric_id] for query_id in query_ids),
                                8,
                            ),
                            "primary": metric_id == "ndcg@10",
                        }
                    )
            groups.append(
                {
                    "id": value_id,
                    "label": value_label,
                    "count": len(query_ids),
                    "results": results,
                }
            )
        breakdowns.append(
            {
                "id": field,
                "label": label,
                "category": "query",
                "description": f"Scores for queries grouped by {label.lower()}.",
                "primary_metric": "ndcg@10",
                **context,
                "values": groups,
            }
        )

    for field, label in RELEVANT_DOCUMENT_BREAKDOWNS:
        raw_values = {str(row["dimensions"][field]) for row in document_rows}
        if field == "language":
            values = [(value, {"zh": "Chinese", "en": "English"}[value]) for value in ("zh", "en")]
        else:
            values = [(value, _label(value)) for value in sorted(raw_values)]
        groups = []
        for value_id, value_label in values:
            targets = {
                query_id: {
                    document_id
                    for document_id in relevant_ids
                    if str(dimensions_by_document[document_id][field]) == value_id
                }
                for query_id, relevant_ids in relevant_by_query.items()
            }
            targets = {query_id: ids for query_id, ids in targets.items() if ids}
            results = []
            for solution_id, rankings in artifact.rankings.items():
                for cutoff in (10, 1, 5):
                    values_at_cutoff = [
                        _target_recall(rankings[query_id], target_ids, cutoff)
                        for query_id, target_ids in targets.items()
                    ]
                    results.append(
                        {
                            "solution_id": solution_id,
                            "metric_id": f"target_recall@{cutoff}",
                            "metric_label": f"Target Recall@{cutoff}",
                            "value": round(statistics.mean(values_at_cutoff), 8),
                            "primary": cutoff == 10,
                        }
                    )
            groups.append(
                {
                    "id": value_id,
                    "label": value_label,
                    "count": len(targets),
                    "target_count": sum(len(ids) for ids in targets.values()),
                    "results": results,
                }
            )
        breakdowns.append(
            {
                "id": f"relevant_document_{field}",
                "label": label,
                "category": "relevant_documents",
                "description": (
                    "Recall of relevant documents in this group within each existing Top-10; retrieval is not rerun."
                ),
                "primary_metric": "target_recall@10",
                **context,
                "values": groups,
            }
        )
    return breakdowns
