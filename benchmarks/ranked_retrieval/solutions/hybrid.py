"""Hybrid Ranked Retrieval Solution definitions."""

from benchmarks.ranked_retrieval.definition import DATASET_REVISION
from benchmarks.ranked_retrieval.solutions.bm25 import build_bm25_chunk_component
from benchmarks.ranked_retrieval.solutions.qwen import build_qwen_8b_chunk_component
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ReciprocalRankFusionSolution, WeightedScoreFusionSolution

CANDIDATE_CACHE_NAMESPACE = f"ranked:{DATASET_REVISION}:bm25-qwen3-embedding-8b-chunks:v1"


def _components(runtime: Runtime):
    return (
        build_bm25_chunk_component(
            runtime,
            solution_id="internal-bm25-chunk",
            title="Internal BM25 chunks",
            description="Retrieve shared text chunks with BM25.",
            collection_prefix="modern_ir_internal_bm25_chunk",
        ),
        build_qwen_8b_chunk_component(
            runtime,
            solution_id="internal-qwen3-embedding-8b-chunk",
            title="Internal Qwen3 Embedding 8B chunks",
            description="Retrieve shared text chunks with Qwen3 Embedding 8B.",
        ),
    )


def _build_bm25_qwen_rrf(runtime: Runtime, spec: SolutionSpec):
    return ReciprocalRankFusionSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        components=_components(runtime),
        candidate_depth=50,
        rank_constant=60,
        candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
        candidate_cache_namespace=CANDIDATE_CACHE_NAMESPACE,
    )


def _build_weighted(runtime: Runtime, spec: SolutionSpec, *, weights: tuple[float, float]):
    return WeightedScoreFusionSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        components=_components(runtime),
        weights=weights,
        score_directions=("lower", "higher"),
        candidate_depth=50,
        candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
        candidate_cache_namespace=CANDIDATE_CACHE_NAMESPACE,
    )


def _build_weighted_25_75(runtime: Runtime, spec: SolutionSpec):
    return _build_weighted(runtime, spec, weights=(0.25, 0.75))


def _build_weighted_50_50(runtime: Runtime, spec: SolutionSpec):
    return _build_weighted(runtime, spec, weights=(0.50, 0.50))


def _build_weighted_75_25(runtime: Runtime, spec: SolutionSpec):
    return _build_weighted(runtime, spec, weights=(0.75, 0.25))


SPECS = (
    SolutionSpec(
        "bm25-qwen3-embedding-8b-rrf",
        "BM25 + Qwen3 Embedding 8B · RRF",
        "Fuse BM25 and Qwen3 Embedding 8B Top-50 document rankings with equal-weight RRF.",
        ("lexical", "dense", "hybrid", "managed-api", "chunked", "rrf"),
        _build_bm25_qwen_rrf,
        "docs/solutions/bm25-qwen3-embedding-8b-rrf.md",
    ),
    SolutionSpec(
        "bm25-qwen3-embedding-8b-weighted-25-75",
        "BM25 + Qwen3 Embedding 8B · Weighted 25/75",
        "Fuse min-max normalized BM25 and Qwen3 Embedding 8B Top-50 scores with 25/75 weights.",
        ("lexical", "dense", "hybrid", "managed-api", "chunked", "weighted"),
        _build_weighted_25_75,
        "docs/solutions/bm25-qwen3-embedding-8b-weighted.md",
    ),
    SolutionSpec(
        "bm25-qwen3-embedding-8b-weighted-50-50",
        "BM25 + Qwen3 Embedding 8B · Weighted 50/50",
        "Fuse min-max normalized BM25 and Qwen3 Embedding 8B Top-50 scores with equal weights.",
        ("lexical", "dense", "hybrid", "managed-api", "chunked", "weighted"),
        _build_weighted_50_50,
        "docs/solutions/bm25-qwen3-embedding-8b-weighted.md",
    ),
    SolutionSpec(
        "bm25-qwen3-embedding-8b-weighted-75-25",
        "BM25 + Qwen3 Embedding 8B · Weighted 75/25",
        "Fuse min-max normalized BM25 and Qwen3 Embedding 8B Top-50 scores with 75/25 weights.",
        ("lexical", "dense", "hybrid", "managed-api", "chunked", "weighted"),
        _build_weighted_75_25,
        "docs/solutions/bm25-qwen3-embedding-8b-weighted.md",
    ),
)
