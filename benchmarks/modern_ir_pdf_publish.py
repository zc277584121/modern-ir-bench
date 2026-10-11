"""Validate the live PDF run, preserve its evidence, and update the Space."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, Pdf, load_dataset

from benchmarks.modern_ir_pdf_definition import (
    DATASET_ID,
    DATASET_REVISION,
    RELEASE_ID,
    TASK_ID,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = PROJECT_ROOT / "artifacts/modern-ir-pdf-live"
RESULT_ROOT = PROJECT_ROOT / "results/modern-ir-pdf-v1"
SPACE_RESULTS = PROJECT_ROOT / "space/data/results.json"
SOLUTION_ID = "poppler-text-layer-bm25-full"
METRIC_ORDER = ("ndcg@10", "recall@1", "recall@5", "recall@10", "mrr@10")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _query_metrics(expected: set[str], ranked: list[str]) -> dict[str, float]:
    recalls = {
        f"recall@{cutoff}": len(expected.intersection(ranked[:cutoff])) / len(expected)
        for cutoff in (1, 5, 10)
    }
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


def _metric_results(
    *,
    metrics_by_query: dict[str, dict[str, float]],
    query_ids: set[str],
) -> list[dict[str, object]]:
    labels = {
        "ndcg@10": "NDCG@10",
        "recall@1": "Recall@1",
        "recall@5": "Recall@5",
        "recall@10": "Recall@10",
        "mrr@10": "MRR@10",
    }
    return [
        {
            "solution_id": SOLUTION_ID,
            "metric_id": metric_id,
            "metric_label": labels[metric_id],
            "value": round(statistics.mean(metrics_by_query[query_id][metric_id] for query_id in query_ids), 8),
            "primary": metric_id == "ndcg@10",
        }
        for metric_id in METRIC_ORDER
    ]


def _query_breakdown(
    *,
    breakdown_id: str,
    label: str,
    description: str,
    groups: list[tuple[str, str, set[str]]],
    metrics_by_query: dict[str, dict[str, float]],
) -> dict[str, object]:
    return {
        "id": breakdown_id,
        "label": label,
        "category": "query",
        "description": description,
        "primary_metric": "ndcg@10",
        "task_id": TASK_ID,
        "dataset_id": RELEASE_ID,
        "dataset_version": DATASET_REVISION[:12],
        "values": [
            {
                "id": value_id,
                "label": value_label,
                "count": len(query_ids),
                "results": _metric_results(metrics_by_query=metrics_by_query, query_ids=query_ids),
            }
            for value_id, value_label, query_ids in groups
        ],
    }


def _target_breakdown(
    *,
    breakdown_id: str,
    label: str,
    description: str,
    values: list[tuple[str, str]],
    field: str,
    document_by_id: dict[str, dict],
    relevant_by_query: dict[str, set[str]],
    ranked_by_query: dict[str, list[str]],
) -> dict[str, object]:
    groups = []
    for value_id, value_label in values:
        targets = {
            query_id: {
                document_id
                for document_id in relevant_ids
                if str(document_by_id[document_id][field]).lower() == value_id
            }
            for query_id, relevant_ids in relevant_by_query.items()
        }
        targets = {query_id: ids for query_id, ids in targets.items() if ids}
        results = []
        for cutoff in (10, 1, 5):
            scores = [
                len(set(ranked_by_query[query_id][:cutoff]).intersection(target_ids)) / len(target_ids)
                for query_id, target_ids in targets.items()
            ]
            results.append(
                {
                    "solution_id": SOLUTION_ID,
                    "metric_id": f"target_recall@{cutoff}",
                    "metric_label": f"Target Recall@{cutoff}",
                    "value": round(statistics.mean(scores), 8),
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
    return {
        "id": breakdown_id,
        "label": label,
        "category": "relevant_documents",
        "description": description,
        "primary_metric": "target_recall@10",
        "task_id": TASK_ID,
        "dataset_id": RELEASE_ID,
        "dataset_version": DATASET_REVISION[:12],
        "values": groups,
    }


def _build_breakdowns(observations: Dataset) -> list[dict[str, object]]:
    documents = load_dataset(DATASET_ID, "documents", split="documents", revision=DATASET_REVISION)
    documents = documents.cast_column("pdf", Pdf(decode=False))
    queries = load_dataset(DATASET_ID, "queries", split="queries", revision=DATASET_REVISION)
    qrels = load_dataset(DATASET_ID, "qrels", split="qrels", revision=DATASET_REVISION)
    document_by_id = {str(row["document_id"]): row for row in documents.remove_columns("pdf")}
    query_by_id = {str(row["query_id"]): row for row in queries}
    relevant_by_query: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        if int(row["relevance"]) > 0:
            relevant_by_query[str(row["query_id"])].add(str(row["document_id"]))
    ranked_by_query = {str(row["query_id"]): list(row["ranked_ids"]) for row in observations}
    metrics_by_query = {
        query_id: _query_metrics(relevant_by_query[query_id], ranked_ids)
        for query_id, ranked_ids in ranked_by_query.items()
    }
    query_ids = set(query_by_id)
    with_text = {
        query_id
        for query_id, relevant_ids in relevant_by_query.items()
        if any(bool(document_by_id[document_id]["has_text_layer"]) for document_id in relevant_ids)
    }
    return [
        _query_breakdown(
            breakdown_id="language",
            label="Language",
            description="Scores for queries grouped by language.",
            groups=[
                ("zh", "Chinese", {query_id for query_id in query_ids if query_by_id[query_id]["language"] == "zh"}),
                ("en", "English", {query_id for query_id in query_ids if query_by_id[query_id]["language"] == "en"}),
            ],
            metrics_by_query=metrics_by_query,
        ),
        _query_breakdown(
            breakdown_id="relevant_pdf_access",
            label="Relevant PDF access",
            description=(
                "Separates queries with at least one relevant PDF containing an existing text layer "
                "from queries whose relevant PDFs all require OCR or visual understanding."
            ),
            groups=[
                ("text_layer_available", "Text layer available", with_text),
                ("image_only", "Image-only relevant PDFs", query_ids - with_text),
            ],
            metrics_by_query=metrics_by_query,
        ),
        _target_breakdown(
            breakdown_id="relevant_document_text_layer",
            label="Text layer",
            description="Recall of relevant PDFs by text-layer availability; retrieval is not rerun.",
            values=[("true", "Has text layer"), ("false", "Image only")],
            field="has_text_layer",
            document_by_id=document_by_id,
            relevant_by_query=relevant_by_query,
            ranked_by_query=ranked_by_query,
        ),
        _target_breakdown(
            breakdown_id="relevant_document_transform_tier",
            label="Capture difficulty",
            description="Recall of relevant PDFs by rendering and capture difficulty; retrieval is not rerun.",
            values=[("easy", "Easy"), ("medium", "Medium"), ("hard", "Hard")],
            field="transform_tier",
            document_by_id=document_by_id,
            relevant_by_query=relevant_by_query,
            ranked_by_query=ranked_by_query,
        ),
    ]


def _preserve_result(live: dict, observations_path: Path) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    result_path = RESULT_ROOT / "results.json"
    result_path.write_text(json.dumps(live, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preserved_observations = RESULT_ROOT / "observations.parquet"
    shutil.copy2(observations_path, preserved_observations)
    source_commit = str(live["results"][0]["source_commit"])
    manifest = {
        "schema_version": 1,
        "artifact_type": "live-pdf-retrieval-observations",
        "release": RELEASE_ID,
        "dataset": {"repository": DATASET_ID, "revision": DATASET_REVISION},
        "source_commit": source_commit,
        "solution": SOLUTION_ID,
        "queries": 1000,
        "top_k": 10,
        "results_sha256": _sha256(result_path),
        "observations_sha256": _sha256(preserved_observations),
    }
    (RESULT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (RESULT_ROOT / "README.md").write_text(
        """# Modern IR PDF Retrieval v1 result evidence

This directory preserves the first full, live result for the public PDF Retrieval Task.
It is result evidence, not a Dataset and not a Solution implementation.

The baseline extracts only existing PDF text layers with Poppler and runs full-document
Milvus BM25. It deliberately does not use OCR or page-image understanding, so image-only
relevant PDFs expose the baseline's expected failure boundary.

## Files

- `results.json`: aggregate metric records with immutable source provenance.
- `observations.parquet`: 1,000 query-level expected and Top-10 ranked document IDs.
- `manifest.json`: pinned Dataset revision and artifact checksums.
""",
        encoding="utf-8",
    )


def _update_space(live: dict, breakdowns: list[dict[str, object]]) -> None:
    payload = json.loads(SPACE_RESULTS.read_text(encoding="utf-8"))
    payload["tasks"] = [task for task in payload["tasks"] if task["id"] != TASK_ID]
    payload["tasks"].append(
        {
            "id": TASK_ID,
            "title": "Modern IR PDF Retrieval v1",
            "description": (
                "Retrieve whole PDFs across generated layouts, scans, screenshots, annotations, "
                "compression, and camera-capture conditions."
            ),
            "version": "1.0.0",
            "primary_metric": "ndcg@10",
        }
    )
    payload["tasks"].sort(key=lambda item: item["id"])
    payload["solutions"] = [solution for solution in payload["solutions"] if solution["id"] != SOLUTION_ID]
    source_commit = str(live["results"][0]["source_commit"])
    source_root = f"https://github.com/zc277584121/modern-ir-bench/blob/{source_commit}"
    payload["solutions"].append(
        {
            **live["solutions"][0],
            "tags": ["lexical", "full-document", "pdf-text-layer", "milvus"],
            "code_url": f"{source_root}/benchmarks/pdf_retrieval_solutions/poppler_bm25.py#L10",
            "readme_url": f"{source_root}/docs/solutions/pdf-text-layer-bm25.md",
            "result_url": (
                "https://github.com/zc277584121/modern-ir-bench/tree/"
                "codex/improve-space-ux/results/modern-ir-pdf-v1"
            ),
        }
    )
    payload["solutions"].sort(key=lambda item: item["id"])
    payload["results"] = [row for row in payload["results"] if row["task_id"] != TASK_ID]
    payload["results"].extend(live["results"])
    payload["breakdowns"] = [row for row in payload.get("breakdowns", []) if row["task_id"] != TASK_ID]
    payload["breakdowns"].extend(breakdowns)
    payload["benchmark"].update(
        {
            "release": "modern-ir-bench-public-20261011",
            "notice": (
                "Ranked Retrieval v2.2 is recomputed from reviewed historical Top-10 rankings. "
                "PDF Retrieval v1 is a live run from pinned public code and data."
            ),
        }
    )
    SPACE_RESULTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    live = json.loads((LIVE_ROOT / "results.json").read_text(encoding="utf-8"))
    observation_files = sorted((LIVE_ROOT / "observations").glob("*.parquet"))
    if len(observation_files) != 1:
        raise ValueError("Expected exactly one PDF observation file")
    observations = Dataset.from_parquet(str(observation_files[0]))
    if len(observations) != 1000 or set(map(len, observations["ranked_ids"])) != {10}:
        raise ValueError("PDF result must contain 1,000 complete Top-10 rankings")
    if {row["source_commit"] for row in live["results"]} != {"1fdc17e61f659910826d3a6836e61930ded77667"}:
        raise ValueError("Unexpected PDF result source commit")
    breakdowns = _build_breakdowns(observations)
    _preserve_result(live, observation_files[0])
    _update_space(live, breakdowns)


if __name__ == "__main__":
    main()
