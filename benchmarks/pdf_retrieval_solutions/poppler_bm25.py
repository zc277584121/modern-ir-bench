"""Text-layer-only PDF retrieval baseline."""

from __future__ import annotations

from modern_ir_bench.retrieval import MilvusBM25Solution
from modern_ir_bench.retrieval.indexes.milvus import MilvusServer
from modern_ir_bench.solutions import PdfTextExtractionSolution


def build_poppler_bm25(target: MilvusServer) -> PdfTextExtractionSolution:
    """Extract existing PDF text layers and index complete documents with BM25."""

    return PdfTextExtractionSolution(
        id="poppler-text-layer-bm25-full",
        title="Poppler text layer + BM25 · full document",
        description="Extract existing PDF text layers with Poppler, then run language-aware Milvus BM25.",
        text_solution=MilvusBM25Solution(
            id="internal-bm25-full",
            title="Internal BM25 full document",
            target=target,
            candidate_multiplier=2,
            collection_prefix="modern_ir_pdf_poppler_bm25",
        ),
    )
