"""Lightweight bilingual OCR baseline for PDF Retrieval."""

from __future__ import annotations

import inspect
from pathlib import Path

from modern_ir_bench.retrieval import MilvusBM25Solution
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget
from modern_ir_bench.solutions import PaddleOcrExtractor, PdfOcrRetrievalSolution

SOLUTION_ID = "paddleocr-v5-mobile-bm25-full"
SOLUTION_TITLE = "PP-OCRv5 Mobile + BM25 · full document"
SOLUTION_DESCRIPTION = "OCR every PDF page with a small bilingual model, then run language-aware Milvus BM25."
README_PATH = "docs/solutions/paddleocr-v5-mobile-bm25.md"


def build_solution(
    target: MilvusTarget,
    *,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> PdfOcrRetrievalSolution:
    """OCR every PDF page, then index each complete document with BM25."""

    return PdfOcrRetrievalSolution(
        id=SOLUTION_ID,
        title=SOLUTION_TITLE,
        description=SOLUTION_DESCRIPTION,
        extractor=PaddleOcrExtractor(device=device, cache_dir=cache_dir),
        text_solution=MilvusBM25Solution(
            id="internal-bm25-full",
            title="Internal BM25 full document",
            target=target,
            candidate_multiplier=2,
            collection_prefix="modern_ir_pdf_paddleocr_v5_mobile_bm25",
        ),
    )


def solution_metadata(repository_url: str, commit: str) -> dict[str, object]:
    """Return presentation metadata without constructing the OCR runtime."""

    source_line = inspect.getsourcelines(build_solution)[1]
    source_root = f"{repository_url}/blob/{commit}"
    return {
        "id": SOLUTION_ID,
        "title": SOLUTION_TITLE,
        "description": SOLUTION_DESCRIPTION,
        "tags": ["ocr", "lexical", "full-document", "local", "milvus"],
        "code_url": f"{source_root}/benchmarks/pdf_retrieval/solution.py#L{source_line}",
        "readme_url": f"{source_root}/{README_PATH}",
    }
