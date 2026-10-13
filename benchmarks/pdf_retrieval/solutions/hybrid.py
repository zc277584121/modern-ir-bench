"""Cross-modal PDF Retrieval Solutions."""

from __future__ import annotations

from benchmarks.pdf_retrieval.definition import DATASET_REVISION
from benchmarks.pdf_retrieval.solutions.ocr import build_bm25_pdf_component
from benchmarks.pdf_retrieval.solutions.types import Runtime, SolutionSpec
from benchmarks.pdf_retrieval.solutions.visual import build_voyage_pages_component
from modern_ir_bench.retrieval import ReciprocalRankFusionSolution, WeightedScoreFusionSolution

CANDIDATE_CACHE_NAMESPACE = f"pdf:{DATASET_REVISION}:ocr-bm25-voyage-multimodal-pages:v1"


def _components(runtime: Runtime):
    return (
        build_bm25_pdf_component(
            runtime,
            solution_id="internal-ocr-bm25-full",
            title="Internal OCR + BM25",
            description="OCR full documents and retrieve with BM25.",
        ),
        build_voyage_pages_component(
            runtime,
            solution_id="internal-voyage-multimodal-pages",
            title="Internal Voyage Multimodal pages",
            description="Render and retrieve PDF pages as images.",
        ),
    )


def _build_bm25_visual_rrf(runtime: Runtime, spec: SolutionSpec) -> ReciprocalRankFusionSolution:
    return ReciprocalRankFusionSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        components=_components(runtime),
        candidate_depth=50,
        rank_constant=60,
        candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
        candidate_cache_namespace=CANDIDATE_CACHE_NAMESPACE,
    )


def _build_bm25_visual_weighted(runtime: Runtime, spec: SolutionSpec) -> WeightedScoreFusionSolution:
    return WeightedScoreFusionSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        components=_components(runtime),
        weights=(0.25, 0.75),
        score_directions=("lower", "higher"),
        candidate_depth=50,
        candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
        candidate_cache_namespace=CANDIDATE_CACHE_NAMESPACE,
    )


SPECS = (
    SolutionSpec(
        id="paddleocr-v5-mobile-bm25-voyage-multimodal-3-5-rrf",
        title="PP-OCRv5 Mobile BM25 + Voyage Multimodal 3.5 · RRF",
        description=(
            "Fuse full-document OCR/BM25 and page-image Voyage Multimodal Top-50 rankings with equal-weight RRF."
        ),
        tags=("ocr", "visual", "hybrid", "managed-api", "page-level", "rrf"),
        factory=_build_bm25_visual_rrf,
        readme_path="docs/solutions/paddleocr-v5-mobile-bm25-voyage-multimodal-3-5-rrf.md",
    ),
    SolutionSpec(
        id="paddleocr-v5-mobile-bm25-voyage-multimodal-3-5-weighted-25-75",
        title="PP-OCRv5 Mobile BM25 + Voyage Multimodal 3.5 · Weighted 25/75",
        description=(
            "Fuse min-max normalized OCR/BM25 and page-image Voyage Multimodal Top-50 scores with 25/75 weights."
        ),
        tags=("ocr", "visual", "hybrid", "managed-api", "page-level", "weighted"),
        factory=_build_bm25_visual_weighted,
        readme_path="docs/solutions/paddleocr-v5-mobile-bm25-voyage-multimodal-3-5-weighted.md",
    ),
)
