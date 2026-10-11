"""Load local or Hub PDF releases into the PdfRetrieval contract."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Pdf, Sequence, Value, load_dataset

from modern_ir_bench.datasets.retrieval_release import load_ranked_retrieval_release
from modern_ir_bench.tasks import PdfRetrieval

PDF_DOCUMENT_FEATURES = Features(
    {
        **PdfRetrieval.dataset_features["documents"],
        "title": Value("string"),
        "language": Value("string"),
        "domain": Value("string"),
        "form": Value("string"),
        "length": Value("string"),
        "surface": Value("string"),
        "layout": Value("string"),
        "transform": Value("string"),
        "transform_tier": Value("string"),
        "difficulty": Sequence(Value("string")),
        "pages": Value("int32"),
        "pdf_bytes": Value("int64"),
        "has_text_layer": Value("bool"),
    }
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_pdf_retrieval_release(
    pdf_run_dir: Path,
    retrieval_release_dir: Path,
    *,
    language: str | None = None,
) -> DatasetDict:
    """Compose generated PDFs with an existing query and Qrels release."""

    pdf_run_dir = pdf_run_dir.resolve()
    ranked = load_ranked_retrieval_release(retrieval_release_dir, language=language)
    rows = _read_jsonl(pdf_run_dir / "manifest.jsonl")
    if language is not None:
        rows = [row for row in rows if row["language"] == language]
    expected_ids = set(ranked["documents"]["document_id"])
    actual_ids = {str(row["document_id"]) for row in rows}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:3]
        extra = sorted(actual_ids - expected_ids)[:3]
        raise ValueError(f"PDF manifest does not match the retrieval corpus: missing={missing}, extra={extra}")
    documents = _pdf_documents(rows, root=pdf_run_dir)
    return DatasetDict(documents=documents, queries=ranked["queries"], qrels=ranked["qrels"])


def load_pdf_retrieval_hub(
    repo_id: str,
    *,
    revision: str,
    language: str | None = None,
) -> DatasetDict:
    """Load a pinned, self-contained PDF retrieval release from the Hub."""

    if not repo_id or not revision:
        raise ValueError("repo_id and revision must be non-empty")
    documents = load_dataset(repo_id, "documents", split="documents", revision=revision)
    queries = load_dataset(repo_id, "queries", split="queries", revision=revision)
    qrels = load_dataset(repo_id, "qrels", split="qrels", revision=revision)
    if documents.features["pdf"] != Pdf(decode=False):
        documents = documents.cast_column("pdf", Pdf(decode=False))
    return _filter_tables(documents=documents, queries=queries, qrels=qrels, language=language)


def _pdf_documents(rows: Iterable[Mapping[str, Any]], *, root: Path) -> Dataset:
    records = list(rows)
    return Dataset.from_dict(
        {
            "document_id": [str(row["document_id"]) for row in records],
            "pdf": [str(root / row["pdf_path"]) for row in records],
            "title": [str(row["title"]) for row in records],
            "language": [str(row["language"]) for row in records],
            "domain": [str(row["domain"]) for row in records],
            "form": [str(row["form"]) for row in records],
            "length": [str(row["length"]) for row in records],
            "surface": [str(row["surface"]) for row in records],
            "layout": [str(row["layout"]) for row in records],
            "transform": [str(row["transform"]) for row in records],
            "transform_tier": [str(row["transform_tier"]) for row in records],
            "difficulty": [list(row["difficulty"]) for row in records],
            "pages": [int(row["pages"]) for row in records],
            "pdf_bytes": [int(row["pdf_bytes"]) for row in records],
            "has_text_layer": [bool(row["has_text_layer"]) for row in records],
        },
        features=PDF_DOCUMENT_FEATURES,
    )


def _filter_tables(
    *,
    documents: Dataset,
    queries: Dataset,
    qrels: Dataset,
    language: str | None,
) -> DatasetDict:
    if language is None:
        return DatasetDict(documents=documents, queries=queries, qrels=qrels)
    documents = documents.filter(lambda row: row["language"] == language, desc=f"Select {language} PDFs")
    queries = queries.filter(lambda row: row["language"] == language, desc=f"Select {language} queries")
    query_ids = set(queries["query_id"])
    qrels = qrels.filter(lambda row: row["query_id"] in query_ids, desc=f"Select {language} Qrels")
    if not documents or not queries:
        raise ValueError(f"release has no documents and queries for language {language!r}")
    return DatasetDict(documents=documents, queries=queries, qrels=qrels)
