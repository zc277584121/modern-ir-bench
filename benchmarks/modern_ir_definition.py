"""The current Modern IR benchmark definition."""

from __future__ import annotations

from modern_ir_bench import MetricSet
from modern_ir_bench.datasets import load_ranked_retrieval_hub
from modern_ir_bench.metrics import NDCG, MeanReciprocalRank, Recall
from modern_ir_bench.retrieval import TextInput
from modern_ir_bench.tasks import RankedRetrieval

DATASET_ID = "zc277584121/modern-ir-bench"
DATASET_REVISION = "c7a38d6aaa3d1bd178952b6be91e12ca4ceb10ff"
RELEASE_ID = "modern-ir-bench-v2.2-20261010"
TASK_ID = "modern-ir-ranked-retrieval-v2.2"


def _document_value(row):
    return TextInput(text=row["content"], language=row["language"])


def _query_value(row):
    return TextInput(text=row["query"], language=row["language"])


def build_task() -> RankedRetrieval:
    """Build one bilingual Task with independent language candidate pools."""

    return RankedRetrieval(
        id=TASK_ID,
        title="Modern IR Ranked Retrieval v2.2",
        description=(
            "Bilingual synthetic retrieval with independent Chinese and English "
            "candidate pools and an expanded binary relevance set."
        ),
        version="2.2.0",
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
        query_batch_size=32,
        document_value=_document_value,
        query_value=_query_value,
        pool_field="language",
    )
