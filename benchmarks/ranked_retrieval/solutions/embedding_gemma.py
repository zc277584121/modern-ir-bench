"""EmbeddingGemma 2 Solution definition."""

from benchmarks.ranked_retrieval.solutions.runtime import text_of
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex


def _build_embedding_gemma_chunk(runtime: Runtime, spec: SolutionSpec):
    return ChunkedDenseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding=runtime.embedding_gemma,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_embedding_gemma_2_chunk",
        ),
        chunker=runtime.chunker,
        candidate_multiplier=3,
        document_input=text_of,
        query_input=text_of,
    )


SPECS = (
    SolutionSpec(
        "embedding-gemma-2-chunk",
        "EmbeddingGemma 2 · 768-token chunks",
        "Pinned local text-only EmbeddingGemma 2 over shared chunks with Milvus FLAT search.",
        ("dense", "self-hosted", "chunked"),
        _build_embedding_gemma_chunk,
        "docs/solutions/embedding-gemma-2.md",
    ),
)
