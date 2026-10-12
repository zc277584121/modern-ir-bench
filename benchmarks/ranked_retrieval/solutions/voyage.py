"""Voyage Solution definitions."""

from benchmarks.ranked_retrieval.solutions.runtime import text_of
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution, DenseRetrievalSolution
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex


def _build_voyage_full(runtime: Runtime, spec: SolutionSpec):
    return DenseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
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


def _build_voyage_chunk(runtime: Runtime, spec: SolutionSpec):
    return ChunkedDenseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
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
        ("dense", "managed-api", "full-document"),
        _build_voyage_full,
        "docs/solutions/voyage-4-large.md",
    ),
    SolutionSpec(
        "voyage-4-large-chunk",
        "Voyage 4 Large · 768-token chunks",
        "Voyage 4 Large embeddings over shared chunks with Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_voyage_chunk,
        "docs/solutions/voyage-4-large.md",
    ),
)
