"""Prepare persistent PDF multi-vector caches without retrieval or evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from benchmarks.languages import select_solutions
from benchmarks.pdf_retrieval.definition import DATASET_REVISION, build_task, dataset_release_id
from benchmarks.pdf_retrieval.solutions import SOLUTION_SPECS, build_solutions
from modern_ir_bench.retrieval import ChunkedMultiVectorRetrievalSolution, RetrievalResource
from modern_ir_bench.retrieval.indexes.milvus import MilvusLite

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", required=True, action="append", choices=[spec.id for spec in SOLUTION_SPECS])
    parser.add_argument("--language", choices=("en", "zh"))
    parser.add_argument("--output", type=Path, help="Optional preparation report.")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ids = select_solutions(SOLUTION_SPECS, arguments.solution, arguments.language)
    solutions = build_solutions(
        target=MilvusLite(PROJECT_ROOT / ".runtime/unused.db"), project_root=PROJECT_ROOT, solution_ids=ids
    )
    if any(not isinstance(solution, ChunkedMultiVectorRetrievalSolution) for solution in solutions):
        parser.error("Preparation supports chunked/page multi-vector Solutions")
    task = build_task(language=arguments.language)
    dataset = task.datasets[dataset_release_id(arguments.language)]
    reports = []
    for solution in solutions:
        report = solution.pre_encode(
            (RetrievalResource(id=row["document_id"], value=task.document_value(row)) for row in dataset["documents"]),
            (task.query_value(row) for row in dataset["queries"]),
        )
        reports.append(
            {
                "solution": solution.id,
                "dataset_revision": DATASET_REVISION,
                "release": dataset_release_id(arguments.language),
                **report,
            }
        )
    content = json.dumps(reports, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
