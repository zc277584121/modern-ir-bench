"""Lightweight bilingual OCR baseline for PDF retrieval."""

from __future__ import annotations

from pathlib import Path

from modern_ir_bench.retrieval import MilvusBM25Solution
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget
from modern_ir_bench.solutions import PaddleOcrExtractor, PdfTextExtractionSolution


def build_paddleocr_bm25(
    target: MilvusTarget,
    *,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> PdfTextExtractionSolution:
    """OCR every PDF page, then index each complete document with BM25."""

    return PdfTextExtractionSolution(
        id="paddleocr-v5-mobile-bm25-full",
        title="PP-OCRv5 Mobile + BM25 · full document",
        description="OCR every PDF page with a small bilingual model, then run language-aware Milvus BM25.",
        extractor=PaddleOcrExtractor(device=device, cache_dir=cache_dir),
        text_solution=MilvusBM25Solution(
            id="internal-bm25-full",
            title="Internal BM25 full document",
            target=target,
            candidate_multiplier=2,
            collection_prefix="modern_ir_pdf_paddleocr_v5_mobile_bm25",
        ),
    )
