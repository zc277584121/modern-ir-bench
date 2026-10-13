"""BGE-M3 dense and learned-sparse Solution definitions."""

from benchmarks.ranked_retrieval.solutions.runtime import text_of
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution, ChunkedSparseRetrievalSolution
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex, MilvusSparseIndex


def build_bge_m3_dense_component(
    runtime: Runtime,
    *,
    solution_id: str,
    title: str,
    description: str,
):
    return ChunkedDenseRetrievalSolution(
        id=solution_id,
        title=title,
        description=description,
        embedding=runtime.bge_m3,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_bge_m3_dense",
        ),
        chunker=runtime.chunker,
        candidate_multiplier=4,
        document_input=text_of,
        query_input=text_of,
    )


def build_bge_m3_sparse_component(
    runtime: Runtime,
    *,
    solution_id: str,
    title: str,
    description: str,
):
    return ChunkedSparseRetrievalSolution(
        id=solution_id,
        title=title,
        description=description,
        embedding=runtime.bge_m3,
        index=MilvusSparseIndex(
            target=runtime.target,
            collection_prefix="modern_ir_bge_m3_sparse",
        ),
        chunker=runtime.chunker,
        candidate_multiplier=4,
        document_input=text_of,
        query_input=text_of,
    )


def _build_bge_m3_dense_chunk(runtime: Runtime, spec: SolutionSpec):
    return build_bge_m3_dense_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
    )


def _build_bge_m3_sparse_chunk(runtime: Runtime, spec: SolutionSpec):
    return build_bge_m3_sparse_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
    )


SPECS = (
    SolutionSpec(
        "bge-m3-dense-chunk",
        "BGE-M3 dense · 768-token chunks",
        "Pinned BGE-M3 dense embeddings over shared chunks with Milvus FLAT search.",
        ("dense", "self-hosted", "chunked"),
        _build_bge_m3_dense_chunk,
        "docs/solutions/bge-m3.md",
    ),
    SolutionSpec(
        "bge-m3-sparse-chunk",
        "BGE-M3 learned sparse · 768-token chunks",
        "Pinned BGE-M3 learned-sparse vectors over shared chunks with Milvus search.",
        ("learned-sparse", "self-hosted", "chunked"),
        _build_bge_m3_sparse_chunk,
        "docs/solutions/bge-m3.md",
    ),
)
