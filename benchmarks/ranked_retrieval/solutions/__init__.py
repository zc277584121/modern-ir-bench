"""Executable Solution catalog for Ranked Retrieval."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from benchmarks.ranked_retrieval.solutions.bge_m3 import SPECS as BGE_M3_SPECS
from benchmarks.ranked_retrieval.solutions.bm25 import SPECS as BM25_SPECS
from benchmarks.ranked_retrieval.solutions.embedding_gemma import SPECS as EMBEDDING_GEMMA_SPECS
from benchmarks.ranked_retrieval.solutions.hybrid import SPECS as HYBRID_SPECS
from benchmarks.ranked_retrieval.solutions.managed import SPECS as MANAGED_SPECS
from benchmarks.ranked_retrieval.solutions.qwen import SPECS as QWEN_SPECS
from benchmarks.ranked_retrieval.solutions.reranking import SPECS as RERANKING_SPECS
from benchmarks.ranked_retrieval.solutions.runtime import build_runtime
from benchmarks.ranked_retrieval.solutions.voyage import SPECS as VOYAGE_SPECS
from benchmarks.solution_docs import load_solution_readme
from modern_ir_bench.core import Solution

SOLUTION_SPECS = (
    *BM25_SPECS,
    *VOYAGE_SPECS,
    *QWEN_SPECS,
    *MANAGED_SPECS,
    *EMBEDDING_GEMMA_SPECS,
    *BGE_M3_SPECS,
    *HYBRID_SPECS,
    *RERANKING_SPECS,
)
SOLUTION_SPECS_BY_ID = {spec.id: spec for spec in SOLUTION_SPECS}


def build_solutions(solution_ids: list[str] | None = None) -> list[Solution]:
    selected = solution_ids or [spec.id for spec in SOLUTION_SPECS]
    unknown = set(selected).difference(SOLUTION_SPECS_BY_ID)
    if unknown:
        raise ValueError(f"Unknown Ranked Retrieval Solution ids: {sorted(unknown)}")
    runtime = build_runtime()
    return [SOLUTION_SPECS_BY_ID[solution_id].build(runtime) for solution_id in selected]


def solution_metadata(repository_url: str, commit: str) -> dict[str, dict[str, Any]]:
    repository_root = Path(__file__).resolve().parents[3]
    metadata = {}
    for spec in SOLUTION_SPECS:
        readme = load_solution_readme(repository_root, spec.readme_path)
        source_file = Path(inspect.getsourcefile(spec.factory) or "").resolve()
        source_path = source_file.relative_to(repository_root)
        source_line = inspect.getsourcelines(spec.factory)[1]
        metadata[spec.id] = {
            "title": spec.title,
            "description": spec.description,
            "tags": list(spec.tags),
            "code_url": f"{repository_url}/blob/{commit}/{source_path}#L{source_line}",
            "readme_url": f"{repository_url}/blob/{commit}/{spec.readme_path}",
            "readme_title": readme.title,
            "readme_preview": readme.preview,
        }
    return metadata
