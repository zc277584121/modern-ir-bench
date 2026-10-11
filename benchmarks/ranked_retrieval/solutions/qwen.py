"""Qwen Solution definition."""

from benchmarks.ranked_retrieval.solutions.runtime import text_of
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex


def _build_qwen_chunk(runtime: Runtime, spec: SolutionSpec):
    return ChunkedDenseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding=runtime.qwen,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_qwen_chunk",
        ),
        chunker=runtime.chunker,
        candidate_multiplier=3,
        document_input=text_of,
        query_input=text_of,
    )


SPECS = (
    SolutionSpec(
        "qwen3-embedding-4b-chunk",
        "Qwen3 Embedding 4B · 768-token chunks",
        "Pinned Qwen3 Embedding 4B over shared chunks with Milvus FLAT search.",
        ("dense", "local", "chunked", "milvus"),
        _build_qwen_chunk,
        "docs/solutions/qwen3-embedding-4b.md",
    ),
)
