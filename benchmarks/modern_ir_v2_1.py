"""Replay and verify Modern IR v2.1 from the pinned public dataset release."""

from __future__ import annotations

import gzip
import inspect
import json
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from datasets import load_dataset

from modern_ir_bench import MetricSet, RunProvenance, RunReport
from modern_ir_bench.datasets import load_ranked_retrieval_hub
from modern_ir_bench.exporters import write_space_results
from modern_ir_bench.metrics import NDCG, MeanReciprocalRank, Recall
from modern_ir_bench.solutions import build_saved_ranking_solutions
from modern_ir_bench.tasks import RankedRetrieval

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/modern-ir-v2.1-retrieval"
SPACE_RESULTS = PROJECT_ROOT / "space/data/results.json"
DATASET_ID = "zc277584121/modern-ir-bench"
DATASET_REVISION = "5bbc4ea34774b89ebec0e57ff82f231ff08d385c"
RELEASE_ID = "modern-ir-bench-v2.1-20261009"
RANKINGS_PATH = PROJECT_ROOT / "results/modern-ir-v2.1/rankings.jsonl.gz"
CHUNK_LABEL = "768-token chunks"
CHUNK_DETAILS = (
    "paragraph-aware chunks targeting 768 tokens with 128-token overlap and the title repeated"
)
SOLUTION_METADATA = {
    "bm25-full": {
        "title": "BM25 · full document",
        "route": "full_bm25",
        "description": "BM25 lexical retrieval over full documents.",
    },
    "bm25-chunk": {
        "title": f"BM25 · {CHUNK_LABEL}",
        "route": "chunk_bm25",
        "description": f"BM25 lexical retrieval over {CHUNK_DETAILS}, merged to document rankings.",
    },
    "voyage-4-large-full": {
        "title": "Voyage 4 Large · full document",
        "route": "full_voyage",
        "description": "Voyage 4 Large dense retrieval over full-document embeddings.",
    },
    "voyage-4-large-chunk": {
        "title": f"Voyage 4 Large · {CHUNK_LABEL}",
        "route": "chunk_voyage",
        "description": (
            f"Voyage 4 Large dense retrieval over {CHUNK_DETAILS}, merged to document rankings."
        ),
    },
    "qwen3-embedding-4b-chunk": {
        "title": f"Qwen3 Embedding 4B · {CHUNK_LABEL}",
        "route": "chunk_qwen",
        "description": (
            f"Qwen3 Embedding 4B dense retrieval over {CHUNK_DETAILS}, merged to document rankings."
        ),
    },
    "bge-m3-dense-chunk": {
        "title": f"BGE-M3 dense · {CHUNK_LABEL}",
        "route": "chunk_bge_m3_dense",
        "description": f"BGE-M3 dense retrieval over {CHUNK_DETAILS}, merged to document rankings.",
    },
    "bge-m3-sparse-chunk": {
        "title": f"BGE-M3 learned sparse · {CHUNK_LABEL}",
        "route": "chunk_bge_m3_sparse",
        "description": (
            f"BGE-M3 learned-sparse retrieval over {CHUNK_DETAILS}, merged to document rankings."
        ),
    },
}


def build_task() -> RankedRetrieval:
    return RankedRetrieval(
        id="modern-ir-ranked-retrieval-v2.1",
        title="Modern IR Ranked Retrieval v2.1",
        description="Bilingual synthetic retrieval over an independently expanded relevance pool.",
        version="2.1.0",
        datasets={
            RELEASE_ID: load_ranked_retrieval_hub(
                DATASET_ID,
                revision=DATASET_REVISION,
            )
        },
        dataset_versions={RELEASE_ID: DATASET_REVISION[:12]},
        metrics=MetricSet(
            primary=NDCG(k=10),
            secondary=[Recall(k=1), Recall(k=5), Recall(k=10), MeanReciprocalRank(k=10)],
        ),
        top_k=10,
        query_batch_size=64,
    )


def load_public_rankings() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries = list(load_dataset(DATASET_ID, "queries", split="queries", revision=DATASET_REVISION))
    with gzip.open(RANKINGS_PATH, "rt", encoding="utf-8") as stream:
        rankings = [json.loads(line) for line in stream if line.strip()]
    return queries, rankings


def metrics_by_solution(records: Iterable[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for record in records:
        metrics.setdefault(str(record["solution_id"]), {})[str(record["metric_id"])] = float(record["value"])
    return metrics


def expected_metrics(rankings: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = {}
    for row in rankings:
        solution = values.setdefault(str(row["solution_id"]), {})
        for metric_id, value in row["metrics"].items():
            solution.setdefault(str(metric_id), []).append(float(value))
    return {
        solution_id: {
            metric_id: statistics.mean(metric_values)
            for metric_id, metric_values in solution.items()
        }
        for solution_id, solution in values.items()
    }


def verify_replay(
    actual: Mapping[str, Mapping[str, float]],
    expected: Mapping[str, Mapping[str, float]],
) -> None:
    if actual.keys() != expected.keys():
        raise RuntimeError("replay Solution set does not match the published rankings")
    for solution_id, values in actual.items():
        for metric_id, value in values.items():
            if abs(value - float(expected[solution_id][metric_id])) > 1e-7:
                raise RuntimeError(
                    f"framework replay mismatch for {solution_id} {metric_id}: "
                    f"{value} != {expected[solution_id][metric_id]}"
                )


def add_solution_code_links(report: RunReport, provenance: RunProvenance) -> None:
    source_lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    for solution_id, solution in report.solutions.items():
        marker = f'    "{solution_id}": {{'
        line = next(
            index
            for index, source_line in enumerate(source_lines, start=1)
            if source_line == marker
        )
        solution["code_url"] = (
            f"{provenance.repository_url}/blob/{provenance.source_commit}/"
            f"benchmarks/modern_ir_v2_1.py#L{line}"
        )


def main() -> None:
    queries, rankings = load_public_rankings()
    solutions = build_saved_ranking_solutions(
        queries=queries,
        rankings=rankings,
        solution_metadata=SOLUTION_METADATA,
    )
    provenance = RunProvenance.capture(
        source_module="benchmarks.modern_ir_v2_1",
        source_path="benchmarks/modern_ir_v2_1.py",
        source_line=inspect.getsourcelines(main)[1],
    )
    report = build_task().run(
        solutions,
        provenance=provenance,
        status="published",
    )
    add_solution_code_links(report, provenance)
    metrics = metrics_by_solution(report.records)
    verify_replay(metrics, expected_metrics(rankings))
    payload = {
        "release": RELEASE_ID,
        "dataset": {
            "repository": DATASET_ID,
            "revision": DATASET_REVISION,
            "documents": 5000,
            "queries": 1000,
            "qrels": 34756,
            "relevant_qrels": 6575,
            "non_relevant_qrels": 28181,
        },
        "metrics_by_solution": metrics,
        "records": report.records,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_space_results(
        report,
        OUTPUT_ROOT / "space-results.json",
        release=RELEASE_ID,
        counts={"documents": 5000, "queries": 1000, "qrels": 34756},
        languages=("Chinese", "English"),
    )
    write_space_results(
        report,
        SPACE_RESULTS,
        release=RELEASE_ID,
        counts={"documents": 5000, "queries": 1000, "qrels": 34756},
        languages=("Chinese", "English"),
    )
    report.write_observations(OUTPUT_ROOT / "observations")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
