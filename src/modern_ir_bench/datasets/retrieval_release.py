"""Load a Modern IR release as the RankedRetrieval Hugging Face contract."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from modern_ir_bench.tasks import RankedRetrieval


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_ranked_retrieval_release(
    release_dir: Path,
    *,
    language: str | None = None,
) -> DatasetDict:
    """Adapt corpus, queries, and Qrels without inventing another Dataset type."""
    release_dir = release_dir.resolve()
    return _adapt_ranked_retrieval_rows(
        documents=_read_jsonl(release_dir / "corpus.jsonl"),
        queries=_read_jsonl(release_dir / "queries.jsonl"),
        qrels=_read_jsonl(release_dir / "qrels.jsonl"),
        language=language,
    )


def load_ranked_retrieval_hub(
    repo_id: str,
    *,
    revision: str,
    language: str | None = None,
) -> DatasetDict:
    """Load a pinned public release from its Hugging Face Dataset repository."""
    if not repo_id or not revision:
        raise ValueError("repo_id and revision must be non-empty")
    return _adapt_ranked_retrieval_rows(
        documents=load_dataset(repo_id, "corpus", split="train", revision=revision),
        queries=load_dataset(repo_id, "queries", split="train", revision=revision),
        qrels=load_dataset(repo_id, "qrels", split="train", revision=revision),
        language=language,
    )


def _adapt_ranked_retrieval_rows(
    *,
    documents: Iterable[Mapping[str, Any]],
    queries: Iterable[Mapping[str, Any]],
    qrels: Iterable[Mapping[str, Any]],
    language: str | None,
) -> DatasetDict:
    documents = list(documents)
    queries = list(queries)
    qrels = list(qrels)
    if language is not None:
        documents = [
            row for row in documents if str(row["dimensions"]["language"]) == language
        ]
        queries = [row for row in queries if str(row["language"]) == language]
        query_ids = {str(row["query_id"]) for row in queries}
        qrels = [row for row in qrels if str(row["query_id"]) in query_ids]
        if not documents or not queries:
            raise ValueError(f"release has no documents and queries for language {language!r}")
    return DatasetDict(
        documents=Dataset.from_dict(
            {
                "document_id": [row["doc_id"] for row in documents],
                "content": [f"{row['title']}\n\n{row['text']}" for row in documents],
            },
            features=RankedRetrieval.dataset_features["documents"],
        ),
        queries=Dataset.from_dict(
            {
                "query_id": [row["query_id"] for row in queries],
                "query": [row["text"] for row in queries],
            },
            features=RankedRetrieval.dataset_features["queries"],
        ),
        qrels=Dataset.from_dict(
            {
                "query_id": [row["query_id"] for row in qrels],
                "document_id": [row["doc_id"] for row in qrels],
                "relevance": [row["relevance"] for row in qrels],
            },
            features=RankedRetrieval.dataset_features["qrels"],
        ),
    )
