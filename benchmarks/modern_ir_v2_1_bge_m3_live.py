"""Run fresh BGE-M3 inference and Milvus retrieval against public v2.1 data."""

from __future__ import annotations

import inspect
import json
import os
import statistics
from pathlib import Path
from typing import Any

from datasets import load_dataset

from benchmarks.modern_ir_v2_1 import (
    DATASET_ID,
    DATASET_REVISION,
    RELEASE_ID,
    expected_metrics,
)
from modern_ir_bench import MetricSet, RunProvenance
from modern_ir_bench.datasets import load_ranked_retrieval_hub
from modern_ir_bench.embeddings import SentenceTransformersEmbedding
from modern_ir_bench.metrics import NDCG, MeanReciprocalRank, Recall
from modern_ir_bench.retrieval import (
    CanonicalTextChunker,
    ChunkedDenseRetrievalSolution,
    HuggingFaceOffsetTokenizer,
)
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex, MilvusServer
from modern_ir_bench.tasks import RankedRetrieval

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/modern-ir-v2.1-bge-m3-live"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
SOLUTION_ID = "bge-m3-dense-chunk-live"


def build_task() -> RankedRetrieval:
    datasets = {
        f"{RELEASE_ID}-{language}": load_ranked_retrieval_hub(
            DATASET_ID,
            revision=DATASET_REVISION,
            language=language,
        )
        for language in ("en", "zh")
    }
    return RankedRetrieval(
        id="modern-ir-ranked-retrieval-v2.1",
        title="Modern IR Ranked Retrieval v2.1",
        description="Live BGE-M3 chunk retrieval over independent English and Chinese pools.",
        version="2.1.0",
        datasets=datasets,
        dataset_versions={
            dataset_id: f"{DATASET_REVISION[:12]}:{dataset_id.rsplit('-', 1)[-1]}"
            for dataset_id in datasets
        },
        metrics=MetricSet(
            primary=NDCG(k=10),
            secondary=[Recall(k=1), Recall(k=5), Recall(k=10), MeanReciprocalRank(k=10)],
        ),
        top_k=10,
        query_batch_size=32,
    )


def build_solution() -> ChunkedDenseRetrievalSolution:
    return ChunkedDenseRetrievalSolution(
        id=SOLUTION_ID,
        title="BGE-M3 dense · canonical chunk · live",
        description="Fresh BGE-M3 inference with canonical chunks and Milvus Server FLAT search.",
        embedding=SentenceTransformersEmbedding(
            model="BAAI/bge-m3",
            revision=MODEL_REVISION,
            batch_size=int(os.environ.get("MIR_BATCH_SIZE", "12")),
            device=os.environ.get("MIR_DEVICE"),
            normalize=True,
        ),
        chunker=CanonicalTextChunker(HuggingFaceOffsetTokenizer()),
        index=MilvusDenseIndex(
            target=MilvusServer(
                os.environ.get("MIR_MILVUS_URI", "http://127.0.0.1:19530"),
                token=os.environ.get("MIR_MILVUS_TOKEN", ""),
            ),
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_v2_1_bge_m3_live",
        ),
        chunk_batch_size=int(os.environ.get("MIR_BATCH_SIZE", "12")),
        candidate_multiplier=4,
    )


def _metrics(report) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for record in report.records:
        output.setdefault(str(record["dataset_id"]), {})[str(record["metric_id"])] = float(
            record["value"]
        )
    return output


def _overall(by_dataset: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        metric_id: statistics.mean(values[metric_id] for values in by_dataset.values())
        for metric_id in next(iter(by_dataset.values()))
    }


def _ranking_comparison(report, published_rankings: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        str(row["query_id"]): list(row["ranked_doc_ids"])
        for row in published_rankings
        if row["solution_id"] == "bge-m3-dense-chunk"
    }
    exact = 0
    overlaps: list[float] = []
    examples: list[dict[str, Any]] = []
    queries = 0
    for observations in report.observations.values():
        for row in observations:
            query_id = str(row["query_id"])
            actual = list(row["ranked_ids"])
            saved = expected[query_id]
            queries += 1
            exact += actual == saved
            overlaps.append(len(set(actual).intersection(saved)) / len(saved))
            if actual != saved and len(examples) < 10:
                examples.append({"query_id": query_id, "live": actual, "saved": saved})
    return {
        "queries": queries,
        "exact_top10_rankings": exact,
        "exact_top10_rate": exact / queries,
        "mean_top10_set_overlap": statistics.mean(overlaps),
        "first_differences": examples,
    }


def main() -> None:
    published_rankings = list(
        load_dataset(DATASET_ID, "rankings", split="train", revision=DATASET_REVISION)
    )
    provenance = RunProvenance.capture(
        source_module="benchmarks.modern_ir_v2_1_bge_m3_live",
        source_path="benchmarks/modern_ir_v2_1_bge_m3_live.py",
        source_line=inspect.getsourcelines(main)[1],
    )
    report = build_task().run(
        [build_solution()],
        provenance=provenance,
        status="validation",
    )
    by_dataset = _metrics(report)
    overall = _overall(by_dataset)
    saved = expected_metrics(published_rankings)["bge-m3-dense-chunk"]
    payload = {
        "notice": "Fresh reproducibility run against the pinned public release.",
        "release": RELEASE_ID,
        "dataset": {"repository": DATASET_ID, "revision": DATASET_REVISION},
        "model": {"id": "BAAI/bge-m3", "revision": MODEL_REVISION},
        "by_language_pool": by_dataset,
        "overall": overall,
        "saved_baseline": saved,
        "metric_delta": {
            metric_id: value - float(saved[metric_id])
            for metric_id, value in overall.items()
        },
        "ranking_comparison": _ranking_comparison(report, published_rankings),
        "records": report.records,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report.write_observations(OUTPUT_ROOT / "observations")
    print(
        json.dumps(
            {key: payload[key] for key in ("overall", "metric_delta", "ranking_comparison")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
