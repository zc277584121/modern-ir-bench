"""Late-interaction PDF Retrieval Solutions."""

from __future__ import annotations

from benchmarks.pdf_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedMultiVectorRetrievalSolution, TextInput
from modern_ir_bench.retrieval.indexes.milvus import MilvusMultiVectorIndex
from modern_ir_bench.solutions import PdfOcrRetrievalSolution


def _text(value: TextInput) -> str:
    return value.text


def _build_mlateon(runtime: Runtime, spec: SolutionSpec) -> PdfOcrRetrievalSolution:
    return PdfOcrRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        extractor=runtime.ocr,
        text_solution=ChunkedMultiVectorRetrievalSolution(
            id="internal-mlateon-maxsim-chunks",
            title="Internal mLateOn MaxSim chunks",
            embedding=runtime.mlateon,
            index=MilvusMultiVectorIndex(
                target=runtime.target,
                max_vectors=4096,
                collection_prefix="modern_ir_pdf_ocr_mlateon_maxsim_chunks",
            ),
            chunker=runtime.text_chunker,
            chunk_batch_size=runtime.mlateon.batch_size,
            candidate_multiplier=4,
            document_input=_text,
            query_input=_text,
        ),
    )


SPECS = (
    SolutionSpec(
        id="paddleocr-v5-mobile-mlateon-maxsim-chunks",
        title="PP-OCRv5 Mobile + mLateOn · chunk MaxSim",
        description=(
            "OCR every page, encode overlapping text chunks as token vectors with multilingual mLateOn, "
            "search with native Milvus MaxSim, then collapse chunk hits to PDF documents."
        ),
        tags=("ocr", "late-interaction", "maxsim", "self-hosted", "chunked", "multilingual"),
        factory=_build_mlateon,
        readme_path="docs/solutions/paddleocr-v5-mobile-mlateon-maxsim.md",
    ),
)
