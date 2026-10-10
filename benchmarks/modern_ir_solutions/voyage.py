"""Voyage Solution definitions."""

from benchmarks.modern_ir_solutions.runtime import text_of
from benchmarks.modern_ir_solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution, DenseRetrievalSolution
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex


def build_voyage_full(runtime: Runtime):
    return DenseRetrievalSolution(
        id="voyage-4-large-full",
        title="Voyage 4 Large · full document",
        description="Voyage 4 Large embeddings with Milvus FLAT cosine search.",
        embedding=runtime.voyage,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_voyage_full",
        ),
        document_input=text_of,
        query_input=text_of,
    )


def build_voyage_chunk(runtime: Runtime):
    return ChunkedDenseRetrievalSolution(
        id="voyage-4-large-chunk",
        title="Voyage 4 Large · 768-token chunks",
        description="Voyage 4 Large embeddings over shared chunks with Milvus FLAT search.",
        embedding=runtime.voyage,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_voyage_chunk",
        ),
        chunker=runtime.chunker,
        candidate_multiplier=3,
        document_input=text_of,
        query_input=text_of,
    )


SPECS = (
    SolutionSpec(
        "voyage-4-large-full",
        "Voyage 4 Large · full document",
        "Voyage 4 Large full-document embeddings with Milvus FLAT search.",
        ("dense", "hosted", "full-document", "milvus"),
        build_voyage_full,
        "docs/solutions/voyage-4-large.md",
    ),
    SolutionSpec(
        "voyage-4-large-chunk",
        "Voyage 4 Large · 768-token chunks",
        "Voyage 4 Large embeddings over shared chunks with Milvus FLAT search.",
        ("dense", "hosted", "chunked", "milvus"),
        build_voyage_chunk,
        "docs/solutions/voyage-4-large.md",
    ),
)
