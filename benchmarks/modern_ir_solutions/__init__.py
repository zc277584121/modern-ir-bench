"""Executable Solution catalog for the current Modern IR benchmark."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from benchmarks.modern_ir_solutions.bge_m3 import SPECS as BGE_M3_SPECS
from benchmarks.modern_ir_solutions.bm25 import SPECS as BM25_SPECS
from benchmarks.modern_ir_solutions.qwen import SPECS as QWEN_SPECS
from benchmarks.modern_ir_solutions.runtime import build_runtime
from benchmarks.modern_ir_solutions.voyage import SPECS as VOYAGE_SPECS
from modern_ir_bench.core import Solution

SOLUTION_SPECS = (*BM25_SPECS, *VOYAGE_SPECS, *QWEN_SPECS, *BGE_M3_SPECS)


def build_solutions() -> list[Solution]:
    runtime = build_runtime()
    return [spec.builder(runtime) for spec in SOLUTION_SPECS]


def solution_metadata(repository_url: str, commit: str) -> dict[str, dict[str, Any]]:
    repository_root = Path(__file__).resolve().parents[2]
    metadata = {}
    for spec in SOLUTION_SPECS:
        source_file = Path(inspect.getsourcefile(spec.builder) or "").resolve()
        source_path = source_file.relative_to(repository_root)
        source_line = inspect.getsourcelines(spec.builder)[1]
        metadata[spec.id] = {
            "title": spec.title,
            "description": spec.description,
            "tags": list(spec.tags),
            "code_url": f"{repository_url}/blob/{commit}/{source_path}#L{source_line}",
            "readme_url": f"{repository_url}/blob/{commit}/{spec.readme_path}",
        }
    return metadata
