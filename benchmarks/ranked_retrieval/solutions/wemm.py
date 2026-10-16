"""WeMM text-retrieval Solution definition."""

from benchmarks.ranked_retrieval.solutions.runtime import text_of
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex


def _build_wemm_chunk(runtime: Runtime, spec: SolutionSpec):
    return ChunkedDenseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding=runtime.wemm,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_wemm_embedding_2b_chunk",
        ),
        chunker=runtime.chunker,
        chunk_batch_size=256,
        candidate_multiplier=3,
        document_input=text_of,
        query_input=text_of,
    )


SPECS = (
    SolutionSpec(
        "wemm-embedding-2b-chunk",
        "WeMM Embedding 2B · 768-token chunks",
        "Pinned local WeMM Embedding 2B text retrieval over shared chunks with Milvus FLAT search.",
        ("dense", "multimodal", "self-hosted", "chunked"),
        _build_wemm_chunk,
        "docs/solutions/wemm-embedding-2b-text.md",
    ),
)
