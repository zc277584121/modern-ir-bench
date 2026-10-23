"""Run the maintained PDF Retrieval Solutions on the public PDF release."""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path

from benchmarks.languages import select_solutions
from benchmarks.pdf_retrieval.definition import DATASET_ID, DATASET_REVISION, build_task, dataset_release_id
from benchmarks.pdf_retrieval.solutions import SOLUTION_SPECS, build_solutions
from benchmarks.ranking_evidence import write_run_artifacts
from modern_ir_bench import RunProvenance
from modern_ir_bench.embeddings import CachedMultiVectorEmbedding
from modern_ir_bench.retrieval.indexes.milvus import MilvusLite, MilvusServer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/modern-ir-pdf-runs"
logger = logging.getLogger(__name__)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PDF Retrieval Solutions")
    parser.add_argument(
        "--solution",
        action="append",
        choices=[spec.id for spec in SOLUTION_SPECS],
        help="Solution to run; repeat to select several. Defaults to Solutions covering the requested scope.",
    )
    parser.add_argument(
        "--language",
        choices=("en", "zh"),
        help="Evaluate one language slice. Required for an English-only Solution such as ColBERTv2.",
    )
    parser.add_argument("--cached-only", action="store_true", help="Require prepared multi-vector caches; never encode.")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    selected_ids = select_solutions(SOLUTION_SPECS, arguments.solution, arguments.language)
    task = build_task(language=arguments.language)
    lite_path = os.environ.get("MIR_MILVUS_LITE_PATH")
    target = (
        MilvusLite(lite_path)
        if lite_path
        else MilvusServer(
            os.environ.get("MIR_MILVUS_URI", "http://127.0.0.1:19530"),
            token=os.environ.get("MIR_MILVUS_TOKEN", ""),
        )
    )
    solutions = build_solutions(
        target=target,
        project_root=PROJECT_ROOT,
        solution_ids=selected_ids,
    )
    if arguments.cached_only:
        for solution in solutions:
            embedding = getattr(solution, "embedding", None)
            if not isinstance(embedding, CachedMultiVectorEmbedding):
                raise TypeError("--cached-only requires cached multi-vector Solutions")
            embedding.cache_only = True
    provenance = RunProvenance.capture(
        source_module="benchmarks.pdf_retrieval.run",
        source_path="benchmarks/pdf_retrieval/run.py",
        source_line=inspect.getsourcelines(main)[1],
    )
    report = task.run(solutions, provenance=provenance, status="live")
    directories = write_run_artifacts(
        report,
        output_root=OUTPUT_ROOT if arguments.language is None else OUTPUT_ROOT / "slices" / arguments.language,
        release=dataset_release_id(arguments.language),
        dataset_repository=DATASET_ID,
        dataset_revision=DATASET_REVISION,
    )
    for directory in directories:
        logger.info("Wrote verified run artifact to %s", directory)


if __name__ == "__main__":
    main()
