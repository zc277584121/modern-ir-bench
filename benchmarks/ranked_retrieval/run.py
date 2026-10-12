"""Run maintained Ranked Retrieval Solutions on the public text release."""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path

from benchmarks.ranked_retrieval.definition import (
    DATASET_ID,
    DATASET_REVISION,
    RELEASE_ID,
    build_task,
)
from benchmarks.ranked_retrieval.solutions import SOLUTION_SPECS, build_solutions
from benchmarks.ranking_evidence import write_run_artifacts
from modern_ir_bench import RunProvenance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/modern-ir-ranked-runs"
logger = logging.getLogger(__name__)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solution",
        action="append",
        choices=[spec.id for spec in SOLUTION_SPECS],
        help="Solution to run; repeat the flag to select several. Defaults to the complete catalog.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    task = build_task()
    provenance = RunProvenance.capture(
        source_module="benchmarks.ranked_retrieval.run",
        source_path="benchmarks/ranked_retrieval/run.py",
        source_line=inspect.getsourcelines(main)[1],
    )
    report = task.run(
        build_solutions(arguments.solution),
        provenance=provenance,
        status="live",
    )
    directories = write_run_artifacts(
        report,
        output_root=OUTPUT_ROOT,
        release=RELEASE_ID,
        dataset_repository=DATASET_ID,
        dataset_revision=DATASET_REVISION,
    )
    for directory in directories:
        logger.info("Wrote verified run artifact to %s", directory)


if __name__ == "__main__":
    main()
