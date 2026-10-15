"""Materialize one Solution's document-level Top-K candidates for offline fusion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from benchmarks.ranked_retrieval.definition import (
    DATASET_REVISION,
    RELEASE_ID,
    _document_value,
    _query_value,
    build_task,
)
from benchmarks.ranked_retrieval.solutions import SOLUTION_SPECS_BY_ID, build_solutions
from modern_ir_bench.retrieval import RetrievalResource
from modern_ir_bench.retrieval.candidate_cache import SqliteCandidateCache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / ".runtime/retrieval-candidates.sqlite3"
logger = logging.getLogger(__name__)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", required=True, choices=sorted(SOLUTION_SPECS_BY_ID))
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.depth < 1 or arguments.batch_size < 1:
        raise ValueError("Depth and batch size must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    task = build_task()
    dataset = task.datasets[RELEASE_ID]
    task.validate_dataset(dataset)
    solution = build_solutions([arguments.solution])[0]
    cache = SqliteCandidateCache(arguments.cache, namespace=arguments.namespace)
    try:
        for pool in sorted(set(dataset["queries"][task.pool_field])):
            documents = dataset["documents"].filter(
                lambda row, value=pool: row[task.pool_field] == value,
                desc=f"Select {pool} document pool",
            )
            queries = dataset["queries"].filter(
                lambda row, value=pool: row[task.pool_field] == value,
                desc=f"Select {pool} query pool",
            )
            resources = [RetrievalResource(id=str(row["document_id"]), value=_document_value(row)) for row in documents]
            corpus_key = cache.corpus_key(resources)
            query_values = [_query_value(row) for row in queries]
            existing = cache.read(
                corpus_key=corpus_key,
                component_id=solution.id,
                queries=query_values,
                depth=arguments.depth,
            )
            if len(existing) == len(queries):
                logger.info("Reused %s/%s candidates for all %s queries", solution.id, pool, len(queries))
                continue
            session = solution.prepare(resources)
            try:
                for start in range(0, len(queries), arguments.batch_size):
                    rows = [queries[index] for index in range(start, min(start + arguments.batch_size, len(queries)))]
                    batch_values = [_query_value(row) for row in rows]
                    cached = cache.read(
                        corpus_key=corpus_key,
                        component_id=solution.id,
                        queries=batch_values,
                        depth=arguments.depth,
                    )
                    missing = [index for index in range(len(rows)) if index not in cached]
                    if missing:
                        values = [batch_values[index] for index in missing]
                        rankings = session.search_batch(values, top_k=arguments.depth)
                        cache.write(
                            corpus_key=corpus_key,
                            component_id=solution.id,
                            queries=values,
                            depth=arguments.depth,
                            rankings=rankings,
                        )
                    logger.info(
                        "%s/%s candidates: %s/%s queries",
                        solution.id,
                        pool,
                        min(start + arguments.batch_size, len(queries)),
                        len(queries),
                    )
            finally:
                session.close()
    finally:
        solution.release()
    logger.info(
        "Cached %s at depth %s for dataset %s@%s in namespace %s",
        solution.id,
        arguments.depth,
        RELEASE_ID,
        DATASET_REVISION[:12],
        arguments.namespace,
    )


if __name__ == "__main__":
    main()
