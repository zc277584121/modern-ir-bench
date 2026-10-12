"""Qwen Solution definition."""

from benchmarks.ranked_retrieval.solutions.runtime import text_of
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex


def build_qwen_chunk_component(
    runtime: Runtime,
    *,
    solution_id: str,
    title: str,
    description: str,
    embedding_key: str,
):
    return ChunkedDenseRetrievalSolution(
        id=solution_id,
        title=title,
        description=description,
        embedding=runtime.qwen[embedding_key],
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix=f"modern_ir_{solution_id.replace('-', '_')}",
        ),
        chunker=runtime.chunker,
        candidate_multiplier=3,
        document_input=text_of,
        query_input=text_of,
    )


def _build_qwen_0_6b_chunk(runtime: Runtime, spec: SolutionSpec):
    return build_qwen_chunk_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding_key="0.6b-local",
    )


def _build_qwen_4b_chunk(runtime: Runtime, spec: SolutionSpec):
    return build_qwen_chunk_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding_key="4b-local",
    )


def _build_qwen_8b_openrouter_chunk(runtime: Runtime, spec: SolutionSpec):
    return build_qwen_chunk_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding_key="8b-openrouter",
    )


def build_qwen_8b_chunk_component(
    runtime: Runtime,
    *,
    solution_id: str,
    title: str,
    description: str,
):
    return build_qwen_chunk_component(
        runtime,
        solution_id=solution_id,
        title=title,
        description=description,
        embedding_key="8b-openrouter",
    )


SPECS = (
    SolutionSpec(
        "qwen3-embedding-0-6b-chunk",
        "Qwen3 Embedding 0.6B · 768-token chunks",
        "Pinned local Qwen3 Embedding 0.6B over shared chunks with Milvus FLAT search.",
        ("dense", "self-hosted", "chunked"),
        _build_qwen_0_6b_chunk,
        "docs/solutions/qwen3-embedding-0-6b.md",
    ),
    SolutionSpec(
        "qwen3-embedding-4b-chunk",
        "Qwen3 Embedding 4B · 768-token chunks",
        "Pinned Qwen3 Embedding 4B over shared chunks with Milvus FLAT search.",
        ("dense", "self-hosted", "chunked"),
        _build_qwen_4b_chunk,
        "docs/solutions/qwen3-embedding-4b.md",
    ),
    SolutionSpec(
        "qwen3-embedding-8b-openrouter-chunk",
        "Qwen3 Embedding 8B · 768-token chunks",
        "OpenRouter-hosted Qwen3 Embedding 8B over shared chunks with Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_qwen_8b_openrouter_chunk,
        "docs/solutions/qwen3-embedding-8b-openrouter.md",
    ),
)
