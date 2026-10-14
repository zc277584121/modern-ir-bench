"""Representative BM25 Solution definitions."""

from collections.abc import Mapping

from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import (
    BM25LanguageConfig,
    MilvusBM25Solution,
    default_bm25_language_configs,
)

LANGUAGE_AWARE_BM25 = {
    "en": BM25LanguageConfig(analyzer={"type": "english"}),
    "zh": BM25LanguageConfig(analyzer={"type": "chinese"}),
}

TUNED_LANGUAGE_AWARE_BM25 = {
    "en": BM25LanguageConfig(analyzer={"type": "english"}, k1=1.8, b=0.75),
    "zh": BM25LanguageConfig(analyzer={"type": "chinese"}, k1=2.0, b=0.5),
}


def _build_bm25_full(runtime: Runtime, spec: SolutionSpec):
    return MilvusBM25Solution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        target=runtime.target,
        language_configs=LANGUAGE_AWARE_BM25,
        candidate_multiplier=2,
        collection_prefix="modern_ir_bm25_full",
    )


def build_bm25_chunk_component(
    runtime: Runtime,
    *,
    solution_id: str,
    title: str,
    description: str,
    collection_prefix: str = "modern_ir_bm25_chunk",
    language_configs: Mapping[str, BM25LanguageConfig] | None = None,
):
    resolved_configs = default_bm25_language_configs() if language_configs is None else language_configs
    return MilvusBM25Solution(
        id=solution_id,
        title=title,
        description=description,
        target=runtime.target,
        chunker=runtime.chunker,
        candidate_multiplier=3,
        collection_prefix=collection_prefix,
        language_configs=resolved_configs,
    )


def _build_bm25_chunk(runtime: Runtime, spec: SolutionSpec):
    return build_bm25_chunk_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
        language_configs=LANGUAGE_AWARE_BM25,
    )


def _build_bm25_chunk_tuned(runtime: Runtime, spec: SolutionSpec):
    return build_bm25_chunk_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
        collection_prefix="modern_ir_bm25_chunk_tuned",
        language_configs=TUNED_LANGUAGE_AWARE_BM25,
    )


SPECS = (
    SolutionSpec(
        "bm25-full",
        "BM25 · full document",
        "Milvus BM25 over full documents with language-aware analyzers and default scoring parameters.",
        ("lexical", "self-hosted", "full-document"),
        _build_bm25_full,
        "docs/solutions/bm25.md",
    ),
    SolutionSpec(
        "bm25-chunk",
        "BM25 · 768-token chunks",
        "Language-aware Milvus BM25 over shared chunks, collapsed to document rankings.",
        ("lexical", "self-hosted", "chunked"),
        _build_bm25_chunk,
        "docs/solutions/bm25.md",
    ),
    SolutionSpec(
        "bm25-chunk-tuned",
        "BM25 · tuned chunks",
        "Chunked language-aware BM25 with fixed per-language k1 and b settings.",
        ("lexical", "self-hosted", "chunked", "tuned"),
        _build_bm25_chunk_tuned,
        "docs/solutions/bm25-tuned.md",
    ),
)
