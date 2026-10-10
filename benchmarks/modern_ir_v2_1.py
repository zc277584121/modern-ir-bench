"""Replay Modern IR v2.1 against its immutable Dataset revision."""

from __future__ import annotations

import inspect
from pathlib import Path

from benchmarks.modern_ir_release import (
    DATASET_ID,
    ReleaseConfig,
    _target_recall,
    expected_metrics,
    replay_release,
)
from benchmarks.modern_ir_release import (
    load_public_rankings as load_release_rankings,
)

__all__ = [
    "DATASET_ID",
    "DATASET_REVISION",
    "RELEASE_ID",
    "_target_recall",
    "expected_metrics",
    "load_public_rankings",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_REVISION = "5bbc4ea34774b89ebec0e57ff82f231ff08d385c"
RELEASE_ID = "modern-ir-bench-v2.1-20261009"
CONFIG = ReleaseConfig(
    version="2.1",
    dataset_revision=DATASET_REVISION,
    release_id=RELEASE_ID,
    rankings_path=PROJECT_ROOT / "results/modern-ir-v2.1/rankings.jsonl.gz",
    output_root=PROJECT_ROOT / "artifacts/modern-ir-v2.1-retrieval",
    query_intent_field="task",
)


def load_public_rankings():
    return load_release_rankings(CONFIG)


def main() -> None:
    replay_release(
        CONFIG,
        source_module="benchmarks.modern_ir_v2_1",
        source_path="benchmarks/modern_ir_v2_1.py",
        source_line=inspect.getsourcelines(main)[1],
    )


if __name__ == "__main__":
    main()
