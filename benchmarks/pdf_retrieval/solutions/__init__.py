"""Executable Solution catalog for PDF Retrieval."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from benchmarks.pdf_retrieval.solutions.ocr import SPECS as OCR_SPECS
from benchmarks.pdf_retrieval.solutions.runtime import build_runtime
from benchmarks.pdf_retrieval.solutions.types import SolutionSpec
from benchmarks.pdf_retrieval.solutions.visual import SPECS as VISUAL_SPECS
from modern_ir_bench.core import Solution
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget

SOLUTION_SPECS = (*OCR_SPECS, *VISUAL_SPECS)
SOLUTION_SPECS_BY_ID = {spec.id: spec for spec in SOLUTION_SPECS}


def build_solutions(
    *,
    target: MilvusTarget,
    project_root: Path,
    solution_ids: list[str] | None = None,
) -> list[Solution]:
    selected = solution_ids or [spec.id for spec in SOLUTION_SPECS]
    unknown = set(selected).difference(SOLUTION_SPECS_BY_ID)
    if unknown:
        raise ValueError(f"Unknown PDF Solution ids: {sorted(unknown)}")
    runtime = build_runtime(target=target, project_root=project_root)
    return [SOLUTION_SPECS_BY_ID[solution_id].build(runtime) for solution_id in selected]


def solution_metadata(repository_url: str, commit: str) -> dict[str, dict[str, Any]]:
    repository_root = Path(__file__).resolve().parents[3]
    metadata = {}
    for spec in SOLUTION_SPECS:
        source_file = Path(inspect.getsourcefile(spec.factory) or "").resolve()
        source_path = source_file.relative_to(repository_root)
        source_line = inspect.getsourcelines(spec.factory)[1]
        metadata[spec.id] = {
            "title": spec.title,
            "description": spec.description,
            "tags": list(spec.tags),
            "code_url": f"{repository_url}/blob/{commit}/{source_path}#L{source_line}",
            "readme_url": f"{repository_url}/blob/{commit}/{spec.readme_path}",
        }
    return metadata


__all__ = ["SOLUTION_SPECS", "SolutionSpec", "build_solutions", "solution_metadata"]
