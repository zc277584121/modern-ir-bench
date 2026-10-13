"""Learned-sparse retrieval Solution definitions."""

from benchmarks.ranked_retrieval.solutions.runtime import text_of
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedSparseRetrievalSolution
from modern_ir_bench.retrieval.indexes.milvus import MilvusSparseIndex


def _build_sparse(runtime: Runtime, spec: SolutionSpec, *, embedding_key: str):
    return ChunkedSparseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding=runtime.sparse[embedding_key],
        index=MilvusSparseIndex(
            target=runtime.target,
            collection_prefix=f"modern_ir_{spec.id.replace('-', '_')}",
        ),
        chunker=runtime.chunker,
        candidate_multiplier=4,
        document_input=text_of,
        query_input=text_of,
    )


def _build_opensearch_multilingual(runtime: Runtime, spec: SolutionSpec):
    return _build_sparse(runtime, spec, embedding_key="opensearch-multilingual-v1")


def _build_gte_multilingual(runtime: Runtime, spec: SolutionSpec):
    return _build_sparse(runtime, spec, embedding_key="gte-multilingual-base")


def _build_splade_v3(runtime: Runtime, spec: SolutionSpec):
    return _build_sparse(runtime, spec, embedding_key="splade-v3-distilbert")


def _build_opensearch_v2_distill(runtime: Runtime, spec: SolutionSpec):
    return _build_sparse(runtime, spec, embedding_key="opensearch-doc-v2-distill")


def _build_granite_30m(runtime: Runtime, spec: SolutionSpec):
    return _build_sparse(runtime, spec, embedding_key="granite-30m-sparse")


SPECS = (
    SolutionSpec(
        "opensearch-multilingual-v1-sparse-chunk",
        "OpenSearch Multilingual v1 sparse · 768-token chunks",
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
        "SPLADE v3 DistilBERT · 768-token chunks",
        "The canonical English SPLADE v3 DistilBERT checkpoint over the shared bilingual corpus.",
        ("learned-sparse", "self-hosted", "chunked", "english-only"),
        _build_splade_v3,
        "docs/solutions/splade-v3-distilbert.md",
    ),
    SolutionSpec(
        "opensearch-doc-v2-distill-sparse-chunk",
        "OpenSearch v2 Distill doc-only sparse · 768-token chunks",
        "A compact English doc-only neural sparse model with static query encoding.",
        ("learned-sparse", "self-hosted", "chunked", "english-only", "inference-free-query"),
        _build_opensearch_v2_distill,
        "docs/solutions/opensearch-doc-v2-distill-sparse.md",
    ),
    SolutionSpec(
        "granite-30m-sparse-chunk",
        "Granite Embedding 30M Sparse · 768-token chunks",
        "IBM's lightweight 30M English sparse bi-encoder over the shared bilingual corpus.",
        ("learned-sparse", "self-hosted", "chunked", "english-only", "lightweight"),
        _build_granite_30m,
        "docs/solutions/granite-30m-sparse.md",
    ),
)
