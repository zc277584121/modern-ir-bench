"""Merge verified Ranked Retrieval runs into the published ranking release."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from benchmarks.ranked_retrieval.definition import (
    DATASET_ID,
    DATASET_REVISION,
    RELEASE_ID,
    build_task,
    dataset_release_id,
)
from benchmarks.ranked_retrieval.solutions import SOLUTION_SPECS
from benchmarks.ranking_evidence import merge_release

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/modern-ir-v2.2"


def build_release(
    *,
    run_directories: list[Path],
    remove_solutions: list[str],
    output: Path | None = None,
    language: str | None = None,
    derive_from: Sequence[Path] = (),
    derive_solutions: list[str] | None = None,
) -> None:
    if derive_from and language is None:
        raise ValueError("--derive-from requires --language")
    merge_release(
        task=build_task(language=language),
        dataset_id=dataset_release_id(language),
        dataset_repository=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        run_directories=run_directories,
        output=output or (DEFAULT_OUTPUT if language is None else DEFAULT_OUTPUT / "slices" / language),
        solution_languages={spec.id: spec.languages for spec in SOLUTION_SPECS},
        remove_solutions=remove_solutions,
        derive_from=derive_from,
        derive_solutions=derive_solutions,
        source_task=build_task() if derive_from else None,
        source_dataset_id=RELEASE_ID if derive_from else None,
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
    parser.add_argument("--language", choices=("en", "zh"))
    parser.add_argument(
        "--derive-from",
        action="append",
        default=[],
        type=Path,
        help="Derive a language slice from a complete run or ranking release without retrieval",
    )
    parser.add_argument("--solution", action="append", help="Solution to derive; defaults to every source Solution")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    build_release(
        run_directories=arguments.run,
        remove_solutions=arguments.remove_solution,
        output=arguments.output,
        language=arguments.language,
        derive_from=arguments.derive_from,
        derive_solutions=arguments.solution,
    )


if __name__ == "__main__":
    main()
