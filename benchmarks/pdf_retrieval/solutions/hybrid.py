"""Cross-modal PDF Retrieval Solutions."""

from __future__ import annotations

from benchmarks.pdf_retrieval.definition import DATASET_REVISION
from benchmarks.pdf_retrieval.solutions.ocr import build_bm25_pdf_component, build_voyage_pdf_component
from benchmarks.pdf_retrieval.solutions.types import Runtime, SolutionSpec
from benchmarks.pdf_retrieval.solutions.visual import build_voyage_pages_component
from modern_ir_bench.retrieval import ReciprocalRankFusionSolution

BM25_VISUAL_CACHE_NAMESPACE = f"pdf:{DATASET_REVISION}:ocr-bm25-voyage-multimodal-pages:v1"
VOYAGE_TEXT_VISUAL_CACHE_NAMESPACE = f"pdf:{DATASET_REVISION}:ocr-voyage-4-large-voyage-multimodal-pages:v1"


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
        candidate_cache_namespace=BM25_VISUAL_CACHE_NAMESPACE,
    )


def build_voyage_text_visual_rrf_component(
    runtime: Runtime,
    *,
    solution_id: str,
    title: str,
    description: str,
) -> ReciprocalRankFusionSolution:
    components = (
        build_voyage_pdf_component(
            runtime,
            solution_id="internal-ocr-voyage-4-large-full",
            title="Internal OCR + Voyage 4 Large",
            description="OCR full documents and retrieve with Voyage 4 Large through the Voyage API.",
        ),
        build_voyage_pages_component(
            runtime,
            solution_id="internal-voyage-multimodal-pages",
            title="Internal Voyage Multimodal pages",
            description="Render and retrieve PDF pages as images.",
        ),
    )
    return ReciprocalRankFusionSolution(
        id=solution_id,
        title=title,
        description=description,
        components=components,
        candidate_depth=50,
        rank_constant=60,
        candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
        candidate_cache_namespace=VOYAGE_TEXT_VISUAL_CACHE_NAMESPACE,
    )


def _build_voyage_text_visual_rrf(runtime: Runtime, spec: SolutionSpec) -> ReciprocalRankFusionSolution:
    return build_voyage_text_visual_rrf_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
    )


SPECS = (
    SolutionSpec(
        id="paddleocr-v5-mobile-bm25-voyage-multimodal-3-5-rrf",
        title="PP-OCRv5 Mobile BM25 + Voyage Multimodal 3.5 · RRF",
        description=(
            "Fuse full-document OCR/BM25 and page-image Voyage Multimodal Top-50 rankings with equal-weight RRF."
        ),
        tags=("ocr", "lexical", "visual", "hybrid", "managed-api", "page-level", "rrf"),
        factory=_build_bm25_visual_rrf,
        readme_path="docs/solutions/paddleocr-v5-mobile-bm25-voyage-multimodal-3-5-rrf.md",
    ),
    SolutionSpec(
        id="paddleocr-v5-mobile-voyage-4-large-voyage-multimodal-3-5-rrf",
        title="OCR Voyage 4 Large + Voyage Multimodal 3.5 · RRF",
        description=(
            "Fuse full-document OCR/Voyage 4 Large and page-image Voyage Multimodal Top-50 rankings "
            "with equal-weight RRF."
        ),
        tags=("ocr", "dense", "visual", "hybrid", "managed-api", "page-level", "rrf"),
        factory=_build_voyage_text_visual_rrf,
        readme_path="docs/solutions/paddleocr-v5-mobile-voyage-4-large-voyage-multimodal-3-5-rrf.md",
    ),
)
