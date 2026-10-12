"""Managed dense embedding Solution definitions."""

from benchmarks.ranked_retrieval.solutions.runtime import text_of
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex


def _build_managed_chunk(runtime: Runtime, spec: SolutionSpec, *, embedding_key: str):
    return ChunkedDenseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding=runtime.managed[embedding_key],
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix=f"modern_ir_{spec.id.replace('-', '_')}",
        ),
        chunker=runtime.chunker,
        candidate_multiplier=3,
        document_input=text_of,
        query_input=text_of,
    )


def _build_gemini(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="gemini-embedding-2")


def _build_perplexity(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="pplx-embed-v1-4b")


SPECS = (
    SolutionSpec(
        "gemini-embedding-2-chunk",
        "Gemini Embedding 2 · 768-token chunks",
        "Gemini Embedding 2 over shared chunks with Milvus FLAT search.",
        ("dense", "managed-api", "chunked", "multimodal"),
        _build_gemini,
        "docs/solutions/gemini-embedding-2.md",
    ),
    SolutionSpec(
        "pplx-embed-v1-4b-chunk",
        "Perplexity Embed V1 4B · 768-token chunks",
        "Perplexity Embed V1 4B over shared chunks with Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_perplexity,
        "docs/solutions/pplx-embed-v1-4b.md",
    ),
)
