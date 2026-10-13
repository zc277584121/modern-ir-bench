"""Hybrid Ranked Retrieval Solution definitions."""

from benchmarks.ranked_retrieval.definition import DATASET_REVISION
from benchmarks.ranked_retrieval.solutions.bm25 import build_bm25_chunk_component
from benchmarks.ranked_retrieval.solutions.qwen import build_qwen_8b_chunk_component
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from benchmarks.ranked_retrieval.solutions.voyage import build_voyage_chunk_component
from modern_ir_bench.retrieval import ReciprocalRankFusionSolution, WeightedScoreFusionSolution

QWEN_CANDIDATE_CACHE_NAMESPACE = f"ranked:{DATASET_REVISION}:bm25-qwen3-embedding-8b-chunks:v1"
VOYAGE_CANDIDATE_CACHE_NAMESPACE = f"ranked:{DATASET_REVISION}:bm25-voyage-4-large-chunks:v1"


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
        candidate_cache_namespace=QWEN_CANDIDATE_CACHE_NAMESPACE,
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
        candidate_cache_namespace=QWEN_CANDIDATE_CACHE_NAMESPACE,
    )


def _build_weighted_25_75(runtime: Runtime, spec: SolutionSpec):
    return _build_weighted(runtime, spec, weights=(0.25, 0.75))


def _build_bm25_voyage_rrf(runtime: Runtime, spec: SolutionSpec):
    components = (
        build_bm25_chunk_component(
            runtime,
            solution_id="internal-bm25-chunk",
            title="Internal BM25 chunks",
            description="Retrieve shared text chunks with BM25.",
            collection_prefix="modern_ir_internal_bm25_chunk",
        ),
        build_voyage_chunk_component(
            runtime,
            solution_id="internal-voyage-4-large-chunk",
            title="Internal Voyage 4 Large chunks",
            description="Retrieve shared text chunks with Voyage 4 Large through the Voyage API.",
        ),
    )
    return ReciprocalRankFusionSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        components=components,
        candidate_depth=50,
        rank_constant=60,
        candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
        candidate_cache_namespace=VOYAGE_CANDIDATE_CACHE_NAMESPACE,
    )


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
        "bm25-voyage-4-large-rrf",
        "BM25 + Voyage 4 Large · RRF",
        "Fuse BM25 and Voyage 4 Large Top-50 document rankings with equal-weight RRF.",
        ("lexical", "dense", "hybrid", "managed-api", "chunked", "rrf"),
        _build_bm25_voyage_rrf,
        "docs/solutions/bm25-voyage-4-large-rrf.md",
    ),
)
