"""Hybrid Ranked Retrieval Solution definitions."""

from benchmarks.ranked_retrieval.solutions.bm25 import build_bm25_chunk_component
from benchmarks.ranked_retrieval.solutions.qwen import build_qwen_8b_chunk_component
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ReciprocalRankFusionSolution


def _build_bm25_qwen_rrf(runtime: Runtime, spec: SolutionSpec):
    return ReciprocalRankFusionSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        components=(
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
        ),
        candidate_depth=50,
        rank_constant=60,
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
)
