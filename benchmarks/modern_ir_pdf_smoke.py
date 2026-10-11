"""Run the first end-to-end PDF retrieval baseline on a deterministic sample."""

from __future__ import annotations

import inspect
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from datasets import DatasetDict

from modern_ir_bench import MetricSet, RunProvenance
from modern_ir_bench.datasets import load_pdf_retrieval_release
from modern_ir_bench.metrics import NDCG, MeanReciprocalRank, Recall
from modern_ir_bench.retrieval import MilvusBM25Solution, TextInput
from modern_ir_bench.retrieval.indexes.milvus import MilvusLite
from modern_ir_bench.solutions import PdfTextExtractionSolution
from modern_ir_bench.tasks import PdfRetrieval

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/pdf-retrieval-smoke"


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} to run the local PDF benchmark")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _smoke_provenance() -> RunProvenance:
    """Keep dirty local validation explicit instead of presenting it as publishable evidence."""

    try:
        return RunProvenance.capture(
            source_module="benchmarks.modern_ir_pdf_smoke",
            source_path="benchmarks/modern_ir_pdf_smoke.py",
            source_line=inspect.getsourcelines(main)[1],
        )
    except RuntimeError as error:
        if "dirty Git worktree" not in str(error):
            raise
        return RunProvenance(
            source_commit="uncommitted-smoke",
            source_module="benchmarks.modern_ir_pdf_smoke",
            source_path="benchmarks/modern_ir_pdf_smoke.py",
            source_line=inspect.getsourcelines(main)[1],
            repository_url="https://github.com/zc277584121/modern-ir-bench",
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def select_sample(
    dataset: DatasetDict,
    *,
    queries_per_language: int = 25,
    documents_per_language: int = 500,
) -> DatasetDict:
    """Keep all positives for balanced queries, then add stable distractors."""

    document_language = dict(zip(dataset["documents"]["document_id"], dataset["documents"]["language"], strict=True))
    has_text_layer = dict(zip(dataset["documents"]["document_id"], dataset["documents"]["has_text_layer"], strict=True))
    positives: dict[str, list[str]] = defaultdict(list)
    for row in dataset["qrels"]:
        if row["relevance"] > 0:
            positives[row["query_id"]].append(row["document_id"])
    selected_queries: list[str] = []
    query_language = dict(zip(dataset["queries"]["query_id"], dataset["queries"]["language"], strict=True))
    for language in sorted(set(query_language.values())):
        candidates = [query_id for query_id in dataset["queries"]["query_id"] if query_language[query_id] == language]
        text_grounded = [
            query_id for query_id in candidates if any(has_text_layer[item] for item in positives[query_id])
        ]
        image_only = [
            query_id for query_id in candidates if not any(has_text_layer[item] for item in positives[query_id])
        ]
        text_count = min(len(text_grounded), queries_per_language // 2)
        chosen = text_grounded[:text_count]
        chosen.extend(image_only[: queries_per_language - len(chosen)])
        if len(chosen) < queries_per_language:
            remaining = [query_id for query_id in candidates if query_id not in set(chosen)]
            chosen.extend(remaining[: queries_per_language - len(chosen)])
        selected_queries.extend(chosen)

    selected_query_ids = set(selected_queries)
    selected_documents = {
        row["document_id"] for row in dataset["qrels"] if row["query_id"] in selected_query_ids and row["relevance"] > 0
    }
    for language in sorted(set(document_language.values())):
        language_documents = [
            document_id
            for document_id in dataset["documents"]["document_id"]
            if document_language[document_id] == language
        ]
        required = {document_id for document_id in selected_documents if document_language[document_id] == language}
        if len(required) > documents_per_language:
            raise ValueError(f"Positive documents exceed the {language} sample budget")
        fillers = [document_id for document_id in language_documents if document_id not in required]
        selected_documents.update(fillers[: documents_per_language - len(required)])

    return DatasetDict(
        documents=dataset["documents"].filter(lambda row: row["document_id"] in selected_documents),
        queries=dataset["queries"].filter(lambda row: row["query_id"] in selected_query_ids),
        qrels=dataset["qrels"].filter(
            lambda row: row["query_id"] in selected_query_ids and row["document_id"] in selected_documents
        ),
    )


def main() -> None:
    full = load_pdf_retrieval_release(
        _required_path("MIR_PDF_RUN_DIR"),
        _required_path("MIR_TEXT_RELEASE_DIR"),
    )
    dataset = select_sample(full)
    task = PdfRetrieval(
        id="modern-ir-pdf-retrieval-v1-smoke",
        title="Modern IR PDF Retrieval v1 · smoke",
        description="Text-layer-only lower bound on a balanced PDF retrieval sample.",
        version="1.0.0-smoke",
        datasets={"modern-ir-pdf-v1-smoke": dataset},
        dataset_versions={"modern-ir-pdf-v1-smoke": "local-20261011"},
        metrics=MetricSet(
            primary=NDCG(k=10),
            secondary=[Recall(k=1), Recall(k=5), Recall(k=10), MeanReciprocalRank(k=10)],
        ),
        top_k=10,
        query_batch_size=16,
        query_value=lambda row: TextInput(text=row["query"], language=row["language"]),
        pool_field="language",
    )
    solution = PdfTextExtractionSolution(
        id="poppler-text-layer-bm25-full",
        title="Poppler text layer + BM25 · full document",
        description="Extract the existing PDF text layer with Poppler, then run language-aware Milvus BM25.",
        text_solution=MilvusBM25Solution(
            id="internal-bm25-full",
            title="Internal BM25 full document",
            target=MilvusLite(OUTPUT_ROOT / "milvus.db"),
            candidate_multiplier=2,
            collection_prefix="modern_ir_pdf_smoke",
        ),
    )
    provenance = _smoke_provenance()
    report = task.run([solution], provenance=provenance, status="smoke")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report.write_observations(OUTPUT_ROOT / "observations")
    payload = {
        "sample": {
            "documents": len(dataset["documents"]),
            "queries": len(dataset["queries"]),
            "qrels": len(dataset["qrels"]),
            "documents_with_text_layer": sum(dataset["documents"]["has_text_layer"]),
        },
        "results": report.records,
    }
    (OUTPUT_ROOT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
