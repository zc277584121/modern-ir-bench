"""Learned-sparse retrieval Solution definitions."""

from collections.abc import Mapping
from typing import Any

from benchmarks.ranked_retrieval.solutions.runtime import text_of
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedSparseRetrievalSolution, TokenTextChunker
from modern_ir_bench.retrieval.indexes.milvus import MilvusSparseIndex


def build_sparse_component(
    runtime: Runtime,
    *,
    solution_id: str,
    title: str,
    description: str,
    embedding_key: str,
    chunker: TokenTextChunker | None = None,
    index_params: Mapping[str, Any] | None = None,
    search_params: Mapping[str, Any] | None = None,
):
    return ChunkedSparseRetrievalSolution(
        id=solution_id,
        title=title,
        description=description,
        embedding=runtime.sparse[embedding_key],
        index=MilvusSparseIndex(
            target=runtime.target,
            index_params=index_params or {},
            search_params=search_params or {},
            collection_prefix=f"modern_ir_{solution_id.replace('-', '_')}",
        ),
        chunker=chunker or runtime.chunker,
        candidate_multiplier=4,
        document_input=text_of,
        query_input=text_of,
    )


def _build_sparse(
    runtime: Runtime,
    spec: SolutionSpec,
    *,
    embedding_key: str,
    chunker: TokenTextChunker | None = None,
    index_params: Mapping[str, Any] | None = None,
    search_params: Mapping[str, Any] | None = None,
):
    return build_sparse_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding_key=embedding_key,
        chunker=chunker,
        index_params=index_params,
        search_params=search_params,
    )


def _build_opensearch_multilingual(runtime: Runtime, spec: SolutionSpec):
    return _build_sparse(
        runtime,
        spec,
        embedding_key="opensearch-multilingual-v1",
        chunker=runtime.short_context_chunker,
    )


def _build_gte_multilingual(runtime: Runtime, spec: SolutionSpec):
    return _build_sparse(runtime, spec, embedding_key="gte-multilingual-base")


def _build_splade_v3(runtime: Runtime, spec: SolutionSpec):
    return _build_sparse(
        runtime,
        spec,
        embedding_key="splade-v3-distilbert",
        chunker=runtime.short_context_chunker,
    )


def _build_opensearch_v2_distill(runtime: Runtime, spec: SolutionSpec):
    return _build_sparse(
        runtime,
        spec,
        embedding_key="opensearch-doc-v2-distill",
        chunker=runtime.short_context_chunker,
    )


def _build_granite_30m(runtime: Runtime, spec: SolutionSpec):
    return _build_sparse(
        runtime,
        spec,
        embedding_key="granite-30m-sparse",
        chunker=runtime.short_context_chunker,
        index_params={"drop_ratio_build": 0.2},
        search_params={"drop_ratio_search": 0.2},
    )


SPECS = (
    SolutionSpec(
        "opensearch-multilingual-v1-sparse-chunk",
        "OpenSearch Multilingual v1 sparse · 384-token chunks",
        "A 160M multilingual learned-sparse model with static query encoding and neural document expansion.",
        ("learned-sparse", "self-hosted", "chunked", "multilingual", "inference-free-query"),
        _build_opensearch_multilingual,
        "docs/solutions/opensearch-multilingual-v1-sparse.md",
    ),
    SolutionSpec(
        "gte-multilingual-base-sparse-chunk",
        "GTE Multilingual Base sparse · 768-token chunks",
        "The learned lexical-weight branch of GTE Multilingual Base over the shared chunks.",
        ("learned-sparse", "self-hosted", "chunked", "multilingual"),
        _build_gte_multilingual,
        "docs/solutions/gte-multilingual-base-sparse.md",
    ),
    SolutionSpec(
        "splade-v3-distilbert-sparse-chunk",
        "SPLADE v3 DistilBERT · 384-token chunks",
        "The canonical English SPLADE v3 DistilBERT checkpoint over the shared bilingual corpus.",
        ("learned-sparse", "self-hosted", "chunked", "english-only"),
        _build_splade_v3,
        "docs/solutions/splade-v3-distilbert.md",
    ),
    SolutionSpec(
        "opensearch-doc-v2-distill-sparse-chunk",
        "OpenSearch v2 Distill doc-only sparse · 384-token chunks",
        "A compact English doc-only neural sparse model with static query encoding.",
        ("learned-sparse", "self-hosted", "chunked", "english-only", "inference-free-query"),
        _build_opensearch_v2_distill,
        "docs/solutions/opensearch-doc-v2-distill-sparse.md",
    ),
    SolutionSpec(
        "granite-30m-sparse-chunk",
        "Granite Embedding 30M Sparse · 384-token chunks",
        "IBM's lightweight 30M English sparse bi-encoder with the official sparse-pruning settings.",
        ("learned-sparse", "self-hosted", "chunked", "english-only", "lightweight"),
        _build_granite_30m,
        "docs/solutions/granite-30m-sparse.md",
    ),
)
