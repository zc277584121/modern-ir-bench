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


def build_bm25_chunk_component(
    runtime: Runtime,
    *,
    solution_id: str,
    title: str,
    description: str,
    collection_prefix: str = "modern_ir_bm25_chunk",
):
    return MilvusBM25Solution(
        id=solution_id,
        title=title,
        description=description,
        target=runtime.target,
        chunker=runtime.chunker,
        candidate_multiplier=3,
        collection_prefix=collection_prefix,
    )


def _build_bm25_chunk(runtime: Runtime, spec: SolutionSpec):
    return build_bm25_chunk_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
    )


SPECS = (
    SolutionSpec(
        "bm25-full",
        "BM25 · full document",
        "Milvus BM25 over full documents with language-specific analyzers.",
        ("lexical", "self-hosted", "full-document"),
        _build_bm25_full,
        "docs/solutions/bm25.md",
    ),
    SolutionSpec(
        "bm25-chunk",
        "BM25 · 768-token chunks",
        "Milvus BM25 over shared chunks, collapsed to document rankings.",
        ("lexical", "self-hosted", "chunked"),
        _build_bm25_chunk,
        "docs/solutions/bm25.md",
    ),
)
