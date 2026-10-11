"""OCR-first PDF Retrieval Solutions."""

from __future__ import annotations

from benchmarks.pdf_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import DenseRetrievalSolution, MilvusBM25Solution, TextInput
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex
from modern_ir_bench.solutions import PdfOcrRetrievalSolution


def _text(value: TextInput) -> str:
    return value.text


def _build_bm25(runtime: Runtime, spec: SolutionSpec) -> PdfOcrRetrievalSolution:
    return PdfOcrRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        extractor=runtime.ocr,
        text_solution=MilvusBM25Solution(
            id="internal-bm25-full",
            title="Internal BM25 full document",
            target=runtime.target,
            candidate_multiplier=2,
            collection_prefix="modern_ir_pdf_ocr_bm25",
        ),
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
        text_solution=DenseRetrievalSolution(
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
        ),
    )


SPECS = (
    SolutionSpec(
        id="paddleocr-v5-mobile-bm25-full",
        title="PP-OCRv5 Mobile + BM25 · full document",
        description="OCR every PDF page with a small bilingual model, then run language-aware Milvus BM25.",
        tags=("ocr", "lexical", "full-document", "local", "milvus"),
        factory=_build_bm25,
        readme_path="docs/solutions/paddleocr-v5-mobile-bm25.md",
    ),
    SolutionSpec(
        id="paddleocr-v5-mobile-bge-m3-full",
        title="PP-OCRv5 Mobile + BGE-M3 · full document",
        description="OCR every page, embed each complete document with local BGE-M3, then search with Milvus FLAT.",
        tags=("ocr", "dense", "local", "full-document", "milvus"),
        factory=_build_bge_m3,
        readme_path="docs/solutions/paddleocr-v5-mobile-bge-m3.md",
    ),
    SolutionSpec(
        id="paddleocr-v5-mobile-voyage-4-large-full",
        title="PP-OCRv5 Mobile + Voyage 4 Large · full document",
        description="OCR every page, embed each complete document with Voyage 4 Large, then search with Milvus FLAT.",
        tags=("ocr", "dense", "hosted", "full-document", "milvus"),
        factory=_build_voyage,
        readme_path="docs/solutions/paddleocr-v5-mobile-voyage-4-large.md",
    ),
)
