"""Qwen Solution definition."""

from benchmarks.modern_ir_solutions.runtime import text_of
from benchmarks.modern_ir_solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex


def build_qwen_chunk(runtime: Runtime):
    return ChunkedDenseRetrievalSolution(
        id="qwen3-embedding-4b-chunk",
        title="Qwen3 Embedding 4B · 768-token chunks",
        description="Pinned Qwen3 Embedding 4B over shared chunks with Milvus FLAT search.",
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
        build_qwen_chunk,
        "docs/solutions/qwen3-embedding-4b.md",
    ),
)
