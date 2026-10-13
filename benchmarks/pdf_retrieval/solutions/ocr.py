"""OCR-first PDF Retrieval Solutions."""

from __future__ import annotations

from benchmarks.pdf_retrieval.definition import DATASET_REVISION
from benchmarks.pdf_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import (
    DenseRetrievalSolution,
    MilvusBM25Solution,
    ReciprocalRankFusionSolution,
    TextInput,
)
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex
from modern_ir_bench.solutions import PdfOcrRetrievalSolution


def _text(value: TextInput) -> str:
    return value.text


def _bm25(runtime: Runtime) -> MilvusBM25Solution:
    return MilvusBM25Solution(
        id="internal-bm25-full",
        title="Internal BM25 full document",
        target=runtime.target,
        candidate_multiplier=2,
        collection_prefix="modern_ir_pdf_ocr_bm25",
    )


def _bge_m3(runtime: Runtime) -> DenseRetrievalSolution:
    return DenseRetrievalSolution(
        id="internal-bge-m3-full",
        title="Internal BGE-M3 full document",
        embedding=runtime.bge_m3,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_pdf_ocr_bge_m3_full",
        ),
        document_input=_text,
        query_input=_text,
    )


def build_bm25_pdf_component(
    runtime: Runtime,
    *,
    solution_id: str,
    title: str,
    description: str,
) -> PdfOcrRetrievalSolution:
    """Build the reusable PDF-to-OCR-to-BM25 retrieval path."""
    return PdfOcrRetrievalSolution(
        id=solution_id,
        title=title,
        description=description,
        extractor=runtime.ocr,
        text_solution=_bm25(runtime),
    )


def _build_bm25(runtime: Runtime, spec: SolutionSpec) -> PdfOcrRetrievalSolution:
    return build_bm25_pdf_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
    )


def _build_voyage(runtime: Runtime, spec: SolutionSpec) -> PdfOcrRetrievalSolution:
    return PdfOcrRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        extractor=runtime.ocr,
        text_solution=DenseRetrievalSolution(
            id="internal-voyage-full",
            title="Internal Voyage full document",
            embedding=runtime.voyage_text,
            index=MilvusDenseIndex(
                target=runtime.target,
                metric="COSINE",
                index_type="FLAT",
                collection_prefix="modern_ir_pdf_ocr_voyage_full",
            ),
            document_input=_text,
            query_input=_text,
        ),
    )


def _build_bge_m3(runtime: Runtime, spec: SolutionSpec) -> PdfOcrRetrievalSolution:
    return PdfOcrRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        extractor=runtime.ocr,
        text_solution=_bge_m3(runtime),
    )


def _build_bm25_bge_m3_rrf(runtime: Runtime, spec: SolutionSpec) -> PdfOcrRetrievalSolution:
    return PdfOcrRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        extractor=runtime.ocr,
        text_solution=ReciprocalRankFusionSolution(
            id="internal-bm25-bge-m3-rrf",
            title="Internal BM25 + BGE-M3 RRF",
            components=(_bm25(runtime), _bge_m3(runtime)),
            candidate_depth=50,
            rank_constant=60,
            candidate_cache_path=runtime.cache_root / "retrieval-candidates.sqlite3",
            candidate_cache_namespace=f"pdf:{DATASET_REVISION}:ocr-bm25-bge-m3-full:v1",
        ),
    )


SPECS = (
    SolutionSpec(
        id="paddleocr-v5-mobile-bm25-full",
        title="PP-OCRv5 Mobile + BM25 · full document",
        description="OCR every PDF page with a small bilingual model, then run language-aware Milvus BM25.",
        tags=("ocr", "lexical", "self-hosted", "full-document"),
        factory=_build_bm25,
        readme_path="docs/solutions/paddleocr-v5-mobile-bm25.md",
    ),
    SolutionSpec(
        id="paddleocr-v5-mobile-bge-m3-full",
        title="PP-OCRv5 Mobile + BGE-M3 · full document",
        description="OCR every page, embed each complete document with local BGE-M3, then search with Milvus FLAT.",
        tags=("ocr", "dense", "self-hosted", "full-document"),
        factory=_build_bge_m3,
        readme_path="docs/solutions/paddleocr-v5-mobile-bge-m3.md",
    ),
    SolutionSpec(
        id="paddleocr-v5-mobile-bm25-bge-m3-rrf",
        title="PP-OCRv5 Mobile + BM25/BGE-M3 · RRF",
        description="OCR every page, fuse full-document BM25 and BGE-M3 Top-50 rankings with equal-weight RRF.",
        tags=("ocr", "hybrid", "self-hosted", "full-document", "rrf"),
        factory=_build_bm25_bge_m3_rrf,
        readme_path="docs/solutions/paddleocr-v5-mobile-bm25-bge-m3-rrf.md",
    ),
    SolutionSpec(
        id="paddleocr-v5-mobile-voyage-4-large-full",
        title="PP-OCRv5 Mobile + Voyage 4 Large · full document",
        description=(
            "OCR every page, embed each complete document with Voyage 4 Large through OpenRouter, "
            "then search with Milvus FLAT."
        ),
        tags=("ocr", "dense", "managed-api", "full-document"),
        factory=_build_voyage,
        readme_path="docs/solutions/paddleocr-v5-mobile-voyage-4-large.md",
    ),
)
