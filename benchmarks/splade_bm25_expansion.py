"""Test whether SPLADE document expansion improves tuned Milvus BM25 on English queries."""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.ranked_retrieval.definition import DATASET_REVISION, RELEASE_ID, build_task
from benchmarks.ranked_retrieval.solutions.bm25 import TUNED_LANGUAGE_AWARE_BM25
from benchmarks.ranked_retrieval.solutions.runtime import SPLADE_V3_FULL_REVISION, build_runtime
from benchmarks.reporting import METRIC_ORDER, retrieval_metrics
from modern_ir_bench.embeddings import OnnxSpladeEmbedding
from modern_ir_bench.retrieval import MilvusBM25Solution, RetrievalResource, TextInput

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TERM_CACHE = PROJECT_ROOT / ".runtime/splade-v3-full-expansion-terms.json"
DEFAULT_OUTPUT = PROJECT_ROOT / ".runtime/splade-bm25-expansion.json"
WORD = re.compile(r"[a-z][a-z-]{2,}")
logger = logging.getLogger(__name__)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term-cache", type=Path, default=DEFAULT_TERM_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expansion-sizes", type=int, nargs="+", default=(16, 32, 64))
    return parser.parse_args()


def _novel_terms(
    embedding: OnnxSpladeEmbedding,
    vector: Mapping[int, float],
    text: str,
    *,
    limit: int,
) -> list[str]:
    ranked_ids = [token_id for token_id, _ in sorted(vector.items(), key=lambda row: (-row[1], row[0]))]
    tokens = embedding.tokens_for_ids(ranked_ids)
    existing = set(WORD.findall(text.lower()))
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        term = token.removeprefix("##").lower()
        if not WORD.fullmatch(term) or term in existing or term in seen:
            continue
        terms.append(term)
        seen.add(term)
        if len(terms) == limit:
            break
    return terms


def _build_term_cache(
    *,
    documents: Sequence[Mapping[str, Any]],
    embedding: OnnxSpladeEmbedding,
    chunker: Any,
    limit: int,
) -> dict[str, list[str]]:
    terms_by_document: dict[str, list[str]] = {}
    for index, row in enumerate(documents, start=1):
        text = str(row["content"])
        vectors = embedding.encode_sparse_documents(chunker(text))
        aggregate: dict[int, float] = {}
        for vector in vectors:
            for token_id, weight in vector.items():
                aggregate[token_id] = max(weight, aggregate.get(token_id, 0.0))
        terms_by_document[str(row["document_id"])] = _novel_terms(
            embedding,
            aggregate,
            text,
            limit=limit,
        )
        if index % 100 == 0 or index == len(documents):
            logger.info("Generated expansion terms for %s/%s documents", index, len(documents))
    return terms_by_document


def _load_or_build_terms(
    *,
    path: Path,
    documents: Sequence[Mapping[str, Any]],
    embedding: OnnxSpladeEmbedding,
    chunker: Any,
    limit: int,
) -> dict[str, list[str]]:
    if path.is_file():
        payload = json.loads(path.read_text())
        if (
            payload.get("dataset_revision") == DATASET_REVISION
            and payload.get("model_revision") == SPLADE_V3_FULL_REVISION
            and int(payload.get("limit", 0)) >= limit
        ):
            logger.info("Reusing expansion terms from %s", path)
            return {str(key): list(value) for key, value in payload["terms"].items()}
    terms = _build_term_cache(
        documents=documents,
        embedding=embedding,
        chunker=chunker,
        limit=limit,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_revision": DATASET_REVISION,
                "model_revision": SPLADE_V3_FULL_REVISION,
                "limit": limit,
                "terms": terms,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return terms


def _evaluate(
    *,
    size: int,
    documents: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    relevant: Mapping[str, set[str]],
    terms: Mapping[str, Sequence[str]],
    runtime: Any,
) -> dict[str, Any]:
    solution = MilvusBM25Solution(
        id=f"splade-expanded-bm25-{size}",
        title=f"SPLADE-expanded BM25 · {size} terms",
        description="Tuned English BM25 with novel SPLADE document expansion terms.",
        target=runtime.target,
        chunker=runtime.chunker,
        language_configs={"en": TUNED_LANGUAGE_AWARE_BM25["en"]},
        candidate_multiplier=3,
        collection_prefix=f"modern_ir_splade_expanded_bm25_{size}",
    )
    resources = []
    for row in documents:
        document_id = str(row["document_id"])
        expansion = " ".join(terms[document_id][:size])
        text = str(row["content"])
        if expansion:
            text = f"{text}\n\n{expansion}"
        resources.append(RetrievalResource(id=document_id, value=TextInput(text=text, language="en")))
    session = solution.prepare(resources)
    rankings: list[list[str]] = []
    try:
        for start in range(0, len(queries), 32):
            rows = queries[start : start + 32]
            hits = session.search_batch(
                [TextInput(text=str(row["query"]), language="en") for row in rows],
                top_k=10,
            )
            rankings.extend([[hit.id for hit in ranking] for ranking in hits])
    finally:
        session.close()
    per_query = [
        retrieval_metrics(relevant[str(row["query_id"])], ranking)
        for row, ranking in zip(queries, rankings, strict=True)
    ]
    return {
        "expansion_terms": size,
        "metrics": {
            metric: round(statistics.mean(values[metric] for values in per_query), 8) for metric in METRIC_ORDER
        },
    }


def main() -> None:
    arguments = _arguments()
    if any(size < 0 for size in arguments.expansion_sizes):
        raise ValueError("Expansion sizes must not be negative")
    sizes = sorted({0, *arguments.expansion_sizes})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    task = build_task()
    dataset = task.datasets[RELEASE_ID]
    documents = [row for row in dataset["documents"] if row["language"] == "en"]
    queries = [row for row in dataset["queries"] if row["language"] == "en"]
    query_ids = {str(row["query_id"]) for row in queries}
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in dataset["qrels"]:
        query_id = str(row["query_id"])
        if query_id in query_ids and int(row["relevance"]) > 0:
            relevant[query_id].add(str(row["document_id"]))

    runtime = build_runtime()
    embedding = runtime.sparse["splade-v3-full"]
    if not isinstance(embedding, OnnxSpladeEmbedding):
        raise TypeError("The SPLADE v3 Full runtime must use OnnxSpladeEmbedding")
    try:
        terms = _load_or_build_terms(
            path=arguments.term_cache,
            documents=documents,
            embedding=embedding,
            chunker=runtime.short_context_chunker,
            limit=max(sizes),
        )
    finally:
        embedding.release()

    results = [
        _evaluate(
            size=size,
            documents=documents,
            queries=queries,
            relevant=relevant,
            terms=terms,
            runtime=runtime,
        )
        for size in sizes
    ]
    payload = {
        "schema_version": 1,
        "dataset_id": RELEASE_ID,
        "dataset_revision": DATASET_REVISION,
        "language": "en",
        "queries": len(queries),
        "documents": len(documents),
        "model": "castorini/splade-v3-onnx",
        "model_revision": SPLADE_V3_FULL_REVISION,
        "term_policy": "highest-weight novel alphabetic WordPieces aggregated by max over short chunks",
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
