"""Replay Modern IR v2.2 against its immutable Dataset revision."""

from __future__ import annotations

import inspect
from pathlib import Path

from benchmarks.modern_ir_release import ReleaseConfig, replay_release

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = ReleaseConfig(
    version="2.2",
    dataset_revision="c7a38d6aaa3d1bd178952b6be91e12ca4ceb10ff",
    release_id="modern-ir-bench-v2.2-20261010",
    rankings_path=PROJECT_ROOT / "results/modern-ir-v2.2/rankings.jsonl.gz",
    output_root=PROJECT_ROOT / "artifacts/modern-ir-v2.2-retrieval",
    query_intent_field="query_intent",
    publish_space=True,
)


def main() -> None:
    replay_release(
        CONFIG,
        source_module="benchmarks.modern_ir_v2_2",
        source_path="benchmarks/modern_ir_v2_2.py",
        source_line=inspect.getsourcelines(main)[1],
    )


if __name__ == "__main__":
    main()
