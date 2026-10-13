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


def _build_openai_large(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="text-embedding-3-large")


def _build_perplexity(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="pplx-embed-v1-4b")


def _build_jina(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="jina-embeddings-v5-text-small")


def _build_dashscope(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="text-embedding-v4")


def _build_nemotron(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="nemotron-3-embed-1b")


def _build_mistral(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="mistral-embed-2312")


def _build_voyage_4(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="voyage-4")


def _build_voyage_4_lite(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="voyage-4-lite")


def _build_openai_small(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="text-embedding-3-small")


def _build_perplexity_small(runtime: Runtime, spec: SolutionSpec):
    return _build_managed_chunk(runtime, spec, embedding_key="pplx-embed-v1-0-6b")


SPECS = (
    SolutionSpec(
        "text-embedding-3-large-chunk",
        "OpenAI Text Embedding 3 Large · 768-token chunks",
        "OpenAI Text Embedding 3 Large over shared chunks with Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_openai_large,
        "docs/solutions/text-embedding-3-large.md",
    ),
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
    SolutionSpec(
        "jina-embeddings-v5-text-small-chunk",
        "Jina Embeddings v5 Text Small · 768-token chunks",
        "Jina Embeddings v5 Text Small over shared chunks with Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_jina,
        "docs/solutions/jina-embeddings-v5-text-small.md",
    ),
    SolutionSpec(
        "dashscope-text-embedding-v4-chunk",
        "DashScope Text Embedding v4 · 768-token chunks",
        "DashScope Text Embedding v4 over shared chunks with native query/document modes and Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_dashscope,
        "docs/solutions/dashscope-text-embedding-v4.md",
    ),
    SolutionSpec(
        "nemotron-3-embed-1b-chunk",
        "Nemotron 3 Embed 1B · 768-token chunks",
        "NVIDIA Nemotron 3 Embed 1B over shared chunks with Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_nemotron,
        "docs/solutions/nemotron-3-embed-1b.md",
    ),
    SolutionSpec(
        "mistral-embed-2312-chunk",
        "Mistral Embed 2312 · 768-token chunks",
        "Mistral Embed 2312 over shared chunks with Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_mistral,
        "docs/solutions/mistral-embed-2312.md",
    ),
    SolutionSpec(
        "voyage-4-chunk",
        "Voyage 4 · 768-token chunks",
        "Voyage 4 over shared chunks through OpenRouter with Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_voyage_4,
        "docs/solutions/voyage-4.md",
    ),
    SolutionSpec(
        "voyage-4-lite-chunk",
        "Voyage 4 Lite · 768-token chunks",
        "Voyage 4 Lite over shared chunks through OpenRouter with Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_voyage_4_lite,
        "docs/solutions/voyage-4-lite.md",
    ),
    SolutionSpec(
        "text-embedding-3-small-chunk",
        "OpenAI Text Embedding 3 Small · 768-token chunks",
        "OpenAI Text Embedding 3 Small over shared chunks through OpenRouter with Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_openai_small,
        "docs/solutions/text-embedding-3-small.md",
    ),
    SolutionSpec(
        "pplx-embed-v1-0-6b-chunk",
        "Perplexity Embed V1 0.6B · 768-token chunks",
        "Perplexity Embed V1 0.6B over shared chunks through OpenRouter with Milvus FLAT search.",
        ("dense", "managed-api", "chunked"),
        _build_perplexity_small,
        "docs/solutions/pplx-embed-v1-0-6b.md",
    ),
)
