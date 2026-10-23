"""Provider-specific decision-model reranking Solutions."""

from __future__ import annotations

import os

from benchmarks.ranked_retrieval.definition import DATASET_REVISION
from benchmarks.ranked_retrieval.solutions.hybrid import VOYAGE_CANDIDATE_CACHE_NAMESPACE
from benchmarks.ranked_retrieval.solutions.reranking import build_reranking_solution, build_voyage_candidates
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.rerankers import (
    OpenRouterGroupedQuestionDecisionReranker,
    OpenRouterPointwiseDecisionReranker,
)

GROUP_CACHE_NAMESPACE = f"ranked:{DATASET_REVISION}:decision-groups:v1"


def _build_pplx(runtime: Runtime, spec: SolutionSpec):
    reranker = OpenRouterPointwiseDecisionReranker(
        model="perplexity/pplx-decider-v1.2",
        request_concurrency=int(os.environ.get("MIR_PPLX_DECISION_CONCURRENCY", "48")),
    )
    return build_reranking_solution(
        runtime,
        spec,
        candidates=build_voyage_candidates(runtime),
        candidate_cache_namespace=VOYAGE_CANDIDATE_CACHE_NAMESPACE,
        reranker=reranker,
    )


def _grouped(runtime: Runtime, spec: SolutionSpec, *, model: str, group_size: int, concurrency_env: str):
    reranker = OpenRouterGroupedQuestionDecisionReranker(
        model=model,
        group_size=group_size,
        request_cache_path=runtime.cache_root / "decision-group-scores.sqlite3",
        request_cache_namespace=GROUP_CACHE_NAMESPACE,
        request_concurrency=int(os.environ.get(concurrency_env, "24")),
    )
    return build_reranking_solution(
        runtime,
        spec,
        candidates=build_voyage_candidates(runtime),
        candidate_cache_namespace=VOYAGE_CANDIDATE_CACHE_NAMESPACE,
        reranker=reranker,
    )


def _build_jev(runtime: Runtime, spec: SolutionSpec):
    return _grouped(
        runtime,
        spec,
        model="typesafe/jev-1.14",
        group_size=50,
        concurrency_env="MIR_JEV_DECISION_CONCURRENCY",
    )


def _build_luna(runtime: Runtime, spec: SolutionSpec):
    return _grouped(
        runtime,
        spec,
        model="openai/gpt-6-luna-decisions",
        group_size=10,
        concurrency_env="MIR_LUNA_DECISION_CONCURRENCY",
    )


SPECS = (
    SolutionSpec(
        "voyage-4-large-pplx-decider-v1-2",
        "Voyage 4 Large → PPLX Decider v1.2",
        "Rerank Voyage 4 Large Top-50 candidates independently with PPLX Decider v1.2.",
        (
            "dense",
            "two-stage",
            "reranker",
            "decision",
            "pointwise",
            "managed-api",
            "chunked-candidates",
            "full-document-rerank",
        ),
        _build_pplx,
        "docs/solutions/pplx-decider-v1-2.md",
        languages=("en", "zh"),
    ),
    SolutionSpec(
        "voyage-4-large-jev-1-14",
        "Voyage 4 Large → Jev 1.14",
        "Rerank Voyage 4 Large Top-50 candidates as isolated questions in one Jev 1.14 request.",
        (
            "dense",
            "two-stage",
            "reranker",
            "decision",
            "grouped-questions",
            "managed-api",
            "chunked-candidates",
            "full-document-rerank",
        ),
        _build_jev,
        "docs/solutions/jev-1-14.md",
        languages=("en", "zh"),
    ),
    SolutionSpec(
        "voyage-4-large-gpt-6-luna-decisions",
        "Voyage 4 Large → GPT-6 Luna Decisions",
        "Rerank Voyage 4 Large Top-50 candidates in groups of ten isolated decision questions.",
        (
            "dense",
            "two-stage",
            "reranker",
            "decision",
            "grouped-questions",
            "managed-api",
            "chunked-candidates",
            "full-document-rerank",
        ),
        _build_luna,
        "docs/solutions/gpt-6-luna-decisions.md",
        languages=("en", "zh"),
    ),
)
