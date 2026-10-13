"""Two-stage retrieve-then-rerank Solution definitions."""

from benchmarks.ranked_retrieval.definition import DATASET_REVISION
from benchmarks.ranked_retrieval.solutions.hybrid import CANDIDATE_CACHE_NAMESPACE, _components
from benchmarks.ranked_retrieval.solutions.qwen import build_qwen_8b_chunk_component
from benchmarks.ranked_retrieval.solutions.runtime import text_of
from benchmarks.ranked_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import RerankingSolution, WeightedScoreFusionSolution

RERANK_CACHE_NAMESPACE = f"ranked:{DATASET_REVISION}:whole-doc-reranking:v1"


def _qwen_candidates(runtime: Runtime):
    return build_qwen_8b_chunk_component(
        runtime,
        solution_id="internal-qwen3-embedding-8b-chunk",
        title="Internal Qwen3 Embedding 8B chunks",
        description="Retrieve shared text chunks with Qwen3 Embedding 8B.",
    )


def _weighted_candidates(runtime: Runtime):
    return WeightedScoreFusionSolution(
        id="internal-bm25-qwen3-embedding-8b-weighted-25-75",
        title="Internal BM25 and Qwen3 weighted candidates",
        description="Generate Top-50 candidates with 25/75 normalized score fusion.",
        components=_components(runtime),
        weights=(0.25, 0.75),
        score_directions=("lower", "higher"),
        candidate_depth=50,
        candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
        candidate_cache_namespace=CANDIDATE_CACHE_NAMESPACE,
    )


def _build(
    runtime: Runtime,
    spec: SolutionSpec,
    *,
    candidate_kind: str,
    reranker_key: str,
):
    candidates = _qwen_candidates(runtime) if candidate_kind == "qwen" else _weighted_candidates(runtime)
    return RerankingSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        candidate_solution=candidates,
        reranker=runtime.rerankers[reranker_key],
        document_input=text_of,
        query_input=text_of,
        candidate_depth=50,
        candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
        candidate_cache_namespace=CANDIDATE_CACHE_NAMESPACE,
        rerank_cache_path=runtime.cache_root / "rerank-scores.sqlite3",
        rerank_cache_namespace=RERANK_CACHE_NAMESPACE,
    )


def _build_qwen_voyage(runtime: Runtime, spec: SolutionSpec):
    return _build(runtime, spec, candidate_kind="qwen", reranker_key="voyage-rerank-3")


def _build_weighted_voyage(runtime: Runtime, spec: SolutionSpec):
    return _build(runtime, spec, candidate_kind="weighted", reranker_key="voyage-rerank-3")


def _build_qwen_bge(runtime: Runtime, spec: SolutionSpec):
    return _build(runtime, spec, candidate_kind="qwen", reranker_key="bge-reranker-v2-m3")


def _build_weighted_bge(runtime: Runtime, spec: SolutionSpec):
    return _build(runtime, spec, candidate_kind="weighted", reranker_key="bge-reranker-v2-m3")


SPECS = (
    SolutionSpec(
        "qwen3-embedding-8b-rerank-3",
        "Qwen3 Embedding 8B → Voyage Rerank 3",
        "Rerank Qwen3 Embedding 8B Top-50 candidates as complete documents with Voyage Rerank 3.",
        ("dense", "two-stage", "reranker", "managed-api", "chunked-candidates", "full-document-rerank"),
        _build_qwen_voyage,
        "docs/solutions/voyage-rerank-3.md",
    ),
    SolutionSpec(
        "bm25-qwen3-embedding-8b-weighted-rerank-3",
        "BM25 + Qwen3 Weighted → Voyage Rerank 3",
        "Rerank weighted BM25/Qwen3 Top-50 candidates as complete documents with Voyage Rerank 3.",
        ("hybrid", "two-stage", "reranker", "managed-api", "chunked-candidates", "full-document-rerank"),
        _build_weighted_voyage,
        "docs/solutions/voyage-rerank-3.md",
    ),
    SolutionSpec(
        "qwen3-embedding-8b-bge-reranker-v2-m3",
        "Qwen3 Embedding 8B → BGE Reranker v2 M3",
        "Rerank Qwen3 Embedding 8B Top-50 candidates as complete documents with BGE Reranker v2 M3.",
        ("dense", "two-stage", "reranker", "self-hosted", "chunked-candidates", "full-document-rerank"),
        _build_qwen_bge,
        "docs/solutions/bge-reranker-v2-m3.md",
    ),
    SolutionSpec(
        "bm25-qwen3-embedding-8b-weighted-bge-reranker-v2-m3",
        "BM25 + Qwen3 Weighted → BGE Reranker v2 M3",
        "Rerank weighted BM25/Qwen3 Top-50 candidates as complete documents with BGE Reranker v2 M3.",
        ("hybrid", "two-stage", "reranker", "self-hosted", "chunked-candidates", "full-document-rerank"),
        _build_weighted_bge,
        "docs/solutions/bge-reranker-v2-m3.md",
    ),
)
