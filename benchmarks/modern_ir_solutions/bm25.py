"""BM25 Solution definitions."""

from benchmarks.modern_ir_solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import MilvusBM25Solution


def build_bm25_full(runtime: Runtime):
    return MilvusBM25Solution(
        id="bm25-full",
        title="BM25 · full document",
        description="Milvus BM25 over full documents with language-specific analyzers.",
        target=runtime.target,
        candidate_multiplier=2,
        collection_prefix="modern_ir_bm25_full",
    )


def build_bm25_chunk(runtime: Runtime):
    return MilvusBM25Solution(
        id="bm25-chunk",
        title="BM25 · 768-token chunks",
        description=(
            "Milvus BM25 over shared 768-token chunks with 128-token overlap, collapsed to document rankings."
        ),
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
        build_bm25_full,
        "docs/solutions/bm25.md",
    ),
    SolutionSpec(
        "bm25-chunk",
        "BM25 · 768-token chunks",
        "Milvus BM25 over shared chunks, collapsed to document rankings.",
        ("lexical", "chunked", "milvus"),
        build_bm25_chunk,
        "docs/solutions/bm25.md",
    ),
)
