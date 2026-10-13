"""Merge verified PDF Retrieval runs into the published ranking release."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.pdf_retrieval.definition import DATASET_ID, DATASET_REVISION, RELEASE_ID, build_task
from benchmarks.ranking_evidence import merge_release

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/modern-ir-pdf-v1"


def build_release(
    *,
    run_directories: list[Path],
    remove_solutions: list[str],
    output: Path = DEFAULT_OUTPUT,
) -> None:
    merge_release(
        task=build_task(),
        dataset_id=RELEASE_ID,
        dataset_repository=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        run_directories=run_directories,
        output=output,
        remove_solutions=remove_solutions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        type=Path,
        help="Verified run artifact directory; repeat to merge several Solutions",
    )
    parser.add_argument(
        "--remove-solution",
        action="append",
        default=[],
        help="Published Solution id to remove; repeat to remove several Solutions",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    build_release(
        run_directories=arguments.run,
        remove_solutions=arguments.remove_solution,
        output=arguments.output,
    )


if __name__ == "__main__":
    main()
