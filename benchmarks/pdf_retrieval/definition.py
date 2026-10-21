"""The public Modern IR PDF Retrieval benchmark definition."""

from __future__ import annotations

from modern_ir_bench import MetricSet
from modern_ir_bench.datasets import load_pdf_retrieval_hub
from modern_ir_bench.metrics import NDCG, MeanReciprocalRank, Recall
from modern_ir_bench.retrieval import TextInput
from modern_ir_bench.tasks import PdfRetrieval

DATASET_ID = "zc277584121/modern-ir-bench-pdf"
DATASET_REVISION = "5ac497e8aad79c9c0843c46eb97bb9900f3b109b"
RELEASE_ID = "modern-ir-bench-pdf-v1-20261011"
TASK_ID = "modern-ir-pdf-retrieval-v1"


def _query_value(row):
    return TextInput(text=row["query"], language=row["language"])


def dataset_release_id(language: str | None = None) -> str:
    if language is not None and language not in {"en", "zh"}:
        raise ValueError(f"Unsupported PDF Dataset language slice: {language}")
    return RELEASE_ID if language is None else f"{RELEASE_ID}-language-{language}"


def build_task(*, language: str | None = None) -> PdfRetrieval:
    """Build document-level retrieval over bilingual rendered PDFs."""

    release_id = dataset_release_id(language)
    return PdfRetrieval(
        id=TASK_ID,
        title="Modern IR PDF Retrieval v1",
        description=(
            "Retrieve whole PDF documents from bilingual queries across generated layouts, "
            "scans, screenshots, annotations, compression, and camera-capture conditions."
        ),
        version="1.0.0",
        datasets={
            release_id: load_pdf_retrieval_hub(
                DATASET_ID,
                revision=DATASET_REVISION,
                language=language,
            )
        },
        dataset_versions={release_id: DATASET_REVISION[:12]},
        metrics=MetricSet(
            primary=NDCG(k=10),
            secondary=[Recall(k=1), Recall(k=5), Recall(k=10), MeanReciprocalRank(k=10)],
        ),
        top_k=10,
        query_batch_size=64,
        query_value=_query_value,
        pool_field="language",
    )
