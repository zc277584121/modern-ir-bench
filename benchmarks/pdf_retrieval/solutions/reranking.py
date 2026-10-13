"""Second-stage reranking for PDF retrieval."""

from __future__ import annotations

from functools import partial

from benchmarks.pdf_retrieval.definition import DATASET_REVISION
from benchmarks.pdf_retrieval.solutions.hybrid import build_voyage_text_visual_rrf_component
from benchmarks.pdf_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import PdfInput, RerankingSolution, TextInput
from modern_ir_bench.solutions import PaddleOcrExtractor

CANDIDATE_CACHE_NAMESPACE = f"pdf:{DATASET_REVISION}:ocr-voyage-visual-reranking-candidates:v1"
RERANK_CACHE_NAMESPACE = f"pdf:{DATASET_REVISION}:ocr-text-reranking:v1"


def _ocr_text(extractor: PaddleOcrExtractor, value: PdfInput) -> str:
    return extractor.extract(value) or "[no extractable PDF text]"


def _query_text(value: TextInput) -> str:
    return value.text


def _build_voyage_reranker(runtime: Runtime, spec: SolutionSpec) -> RerankingSolution:
    candidates = build_voyage_text_visual_rrf_component(
        runtime,
        solution_id="internal-ocr-voyage-visual-rrf-top-50",
        title="Internal OCR Voyage and Voyage Multimodal RRF Top-50",
        description="Fuse the two stored Top-50 routes and retain at most 50 candidates.",
    )
    return RerankingSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        candidate_solution=candidates,
        reranker=runtime.reranker,
        document_input=partial(_ocr_text, runtime.ocr),
        query_input=_query_text,
        candidate_depth=50,
        candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
        candidate_cache_namespace=CANDIDATE_CACHE_NAMESPACE,
        rerank_cache_path=runtime.cache_root / "rerank-scores.sqlite3",
        rerank_cache_namespace=RERANK_CACHE_NAMESPACE,
    )


SPECS = (
    SolutionSpec(
        id="paddleocr-v5-mobile-voyage-4-large-voyage-multimodal-3-5-rerank-3",
        title="OCR Voyage 4 Large + Voyage Multimodal 3.5 → Voyage Rerank 3",
        description=(
            "Fuse OCR Voyage 4 Large and Voyage Multimodal Top-50 rankings, keep at most 50 documents, "
            "then rerank their complete OCR text with Voyage Rerank 3."
        ),
        tags=("ocr", "dense", "visual", "hybrid", "two-stage", "reranker", "managed-api", "rrf"),
        factory=_build_voyage_reranker,
        readme_path="docs/solutions/pdf-voyage-rerank-3.md",
    ),
)
