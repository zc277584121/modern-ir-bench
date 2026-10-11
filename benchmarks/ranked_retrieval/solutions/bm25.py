"""BM25 Solution definitions."""

from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import MilvusBM25Solution


def _build_bm25_full(runtime: Runtime, spec: SolutionSpec):
    return MilvusBM25Solution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        target=runtime.target,
        candidate_multiplier=2,
        collection_prefix="modern_ir_bm25_full",
    )


def _build_bm25_chunk(runtime: Runtime, spec: SolutionSpec):
    return MilvusBM25Solution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        target=runtime.target,
        chunker=runtime.chunker,
        candidate_multiplier=3,
        collection_prefix="modern_ir_bm25_chunk",
    )


SPECS = (
    SolutionSpec(
        "bm25-full",
        "BM25 · full document",
        "Milvus BM25 over full documents with language-specific analyzers.",
        ("lexical", "full-document", "milvus"),
        _build_bm25_full,
        "docs/solutions/bm25.md",
    ),
    SolutionSpec(
        "bm25-chunk",
        "BM25 · 768-token chunks",
        "Milvus BM25 over shared chunks, collapsed to document rankings.",
        ("lexical", "chunked", "milvus"),
        _build_bm25_chunk,
        "docs/solutions/bm25.md",
    ),
)
