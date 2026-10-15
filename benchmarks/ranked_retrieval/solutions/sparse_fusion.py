"""Dense, lexical, and learned-sparse fusion Solution definitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from benchmarks.ranked_retrieval.definition import DATASET_REVISION
from benchmarks.ranked_retrieval.solutions.bge_m3 import (
    build_bge_m3_dense_component,
    build_bge_m3_sparse_component,
)
from benchmarks.ranked_retrieval.solutions.bm25 import (
    TUNED_LANGUAGE_AWARE_BM25,
    build_bm25_chunk_component,
)
from benchmarks.ranked_retrieval.solutions.learned_sparse import build_sparse_component
from benchmarks.ranked_retrieval.solutions.qwen import build_qwen_8b_chunk_component
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.core import Solution
from modern_ir_bench.retrieval import ReciprocalRankFusionSolution, WeightedScoreFusionSolution

Family = Literal["bge-dense-sparse", "gte-qwen", "bm25-gte"]
Factory = Callable[[Runtime, SolutionSpec], Solution]

CANDIDATE_CACHE_NAMESPACE = f"ranked:{DATASET_REVISION}:sparse-fusion:v1"
TUNED_BM25_GTE_CACHE_NAMESPACE = f"ranked:{DATASET_REVISION}:bm25-tuned-gte-sparse-weighted:v1"
CANDIDATE_DEPTH = 50


def _components(runtime: Runtime, family: Family) -> tuple[Solution, Solution]:
    if family == "bge-dense-sparse":
        return (
            build_bge_m3_dense_component(
                runtime,
                solution_id="internal-bge-m3-dense-chunk",
                title="Internal BGE-M3 dense chunks",
                description="Retrieve shared chunks with the BGE-M3 dense branch.",
            ),
            build_bge_m3_sparse_component(
                runtime,
                solution_id="internal-bge-m3-sparse-chunk",
                title="Internal BGE-M3 learned-sparse chunks",
                description="Retrieve shared chunks with the BGE-M3 learned-sparse branch.",
            ),
        )
    if family == "gte-qwen":
        return (
            build_sparse_component(
                runtime,
                solution_id="internal-gte-multilingual-base-sparse-chunk",
                title="Internal GTE Multilingual Base sparse chunks",
                description="Retrieve shared chunks with the GTE learned-sparse branch.",
                embedding_key="gte-multilingual-base",
            ),
            build_qwen_8b_chunk_component(
                runtime,
                solution_id="internal-qwen3-embedding-8b-chunk",
                title="Internal Qwen3 Embedding 8B chunks",
                description="Retrieve shared chunks with Qwen3 Embedding 8B.",
            ),
        )
    if family == "bm25-gte":
        return (
            build_bm25_chunk_component(
                runtime,
                solution_id="internal-bm25-chunk",
                title="Internal BM25 chunks",
                description="Retrieve shared chunks with BM25.",
            ),
            build_sparse_component(
                runtime,
                solution_id="internal-gte-multilingual-base-sparse-chunk",
                title="Internal GTE Multilingual Base sparse chunks",
                description="Retrieve shared chunks with the GTE learned-sparse branch.",
                embedding_key="gte-multilingual-base",
            ),
        )
    raise ValueError(f"Unsupported fusion family: {family}")


def _factory(family: Family) -> Factory:
    def build(runtime: Runtime, spec: SolutionSpec) -> Solution:
        return ReciprocalRankFusionSolution(
            id=spec.id,
            title=spec.title,
            description=spec.description,
            components=_components(runtime, family),
            candidate_depth=CANDIDATE_DEPTH,
            candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
            candidate_cache_namespace=CANDIDATE_CACHE_NAMESPACE,
            rank_constant=60,
        )

    return build


def _build_tuned_bm25_gte_weighted(runtime: Runtime, spec: SolutionSpec) -> Solution:
    return WeightedScoreFusionSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        components=(
            build_bm25_chunk_component(
                runtime,
                solution_id="internal-bm25-tuned-chunk",
                title="Internal tuned BM25 chunks",
                description="Retrieve shared text chunks with tuned language-aware BM25.",
                collection_prefix="modern_ir_internal_bm25_tuned_chunk",
                language_configs=TUNED_LANGUAGE_AWARE_BM25,
            ),
            build_sparse_component(
                runtime,
                solution_id="internal-gte-multilingual-base-sparse-chunk",
                title="Internal GTE Multilingual Base sparse chunks",
                description="Retrieve shared chunks with the GTE learned-sparse branch.",
                embedding_key="gte-multilingual-base",
            ),
        ),
        weights=(0.5, 0.5),
        score_directions=("ranked", "higher"),
        candidate_depth=CANDIDATE_DEPTH,
        candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
        candidate_cache_namespace=TUNED_BM25_GTE_CACHE_NAMESPACE,
    )


SPECS = (
    SolutionSpec(
        "bge-m3-dense-sparse-rrf",
        "BGE-M3 dense + sparse · RRF",
        "Fuse the native BGE-M3 dense and learned-sparse Top-50 rankings with RRF.",
        ("dense", "learned-sparse", "hybrid", "self-hosted", "chunked", "rrf"),
        _factory("bge-dense-sparse"),
        "docs/solutions/bge-m3-dense-sparse-fusion.md",
    ),
    SolutionSpec(
        "gte-sparse-qwen3-embedding-8b-rrf",
        "GTE sparse + Qwen3 Embedding 8B · RRF",
        "Fuse GTE learned-sparse and Qwen3 Embedding 8B Top-50 rankings with RRF.",
        ("learned-sparse", "dense", "hybrid", "managed-api", "chunked", "rrf"),
        _factory("gte-qwen"),
        "docs/solutions/gte-sparse-qwen3-embedding-8b-fusion.md",
    ),
    SolutionSpec(
        "bm25-gte-multilingual-sparse-rrf",
        "BM25 + GTE sparse · RRF",
        "Fuse BM25 and GTE Multilingual Base learned-sparse Top-50 rankings with RRF.",
        ("lexical", "learned-sparse", "hybrid", "self-hosted", "chunked", "rrf"),
        _factory("bm25-gte"),
        "docs/solutions/bm25-learned-sparse-fusion.md",
    ),
    SolutionSpec(
        "bm25-tuned-gte-multilingual-sparse-weighted-50-50",
        "BM25 tuned + GTE sparse · Weighted 50/50",
        "Fuse tuned BM25 and GTE Multilingual Base sparse Top-50 scores with fixed equal weights.",
        ("lexical", "learned-sparse", "hybrid", "self-hosted", "chunked", "weighted", "tuned"),
        _build_tuned_bm25_gte_weighted,
        "docs/solutions/bm25-tuned-weighted-fusion.md",
    ),
)
