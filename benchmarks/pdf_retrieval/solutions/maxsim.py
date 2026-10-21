"""Late-interaction PDF Retrieval Solutions."""

from __future__ import annotations

from collections.abc import Callable

from benchmarks.pdf_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedMultiVectorRetrievalSolution, TextInput, TokenTextChunker
from modern_ir_bench.retrieval.indexes import TorchExactMaxSimIndex
from modern_ir_bench.solutions import PdfOcrRetrievalSolution


def _text(value: TextInput) -> str:
    return value.text


def _build_exact_maxsim(
    runtime: Runtime,
    spec: SolutionSpec,
    *,
    embedding_key: str,
    chunker: Callable[[Runtime], TokenTextChunker],
) -> PdfOcrRetrievalSolution:
    embedding = runtime.late_interaction[embedding_key]
    return PdfOcrRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        extractor=runtime.ocr,
        text_solution=ChunkedMultiVectorRetrievalSolution(
            id=f"internal-{embedding_key}-exact-maxsim-chunks",
            title=f"Internal {embedding_key} exact MaxSim chunks",
            embedding=embedding,
            index=TorchExactMaxSimIndex(device=embedding.device),
            chunker=chunker(runtime),
            chunk_batch_size=embedding.batch_size,
            candidate_multiplier=4,
            document_input=_text,
            query_input=_text,
        ),
    )


def _build_mlateon(runtime: Runtime, spec: SolutionSpec) -> PdfOcrRetrievalSolution:
    return _build_exact_maxsim(
        runtime,
        spec,
        embedding_key="mlateon",
        chunker=lambda value: value.text_chunker,
    )


def _build_colbert_v2(runtime: Runtime, spec: SolutionSpec) -> PdfOcrRetrievalSolution:
    return _build_exact_maxsim(
        runtime,
        spec,
        embedding_key="colbert-v2",
        chunker=lambda value: value.short_text_chunker,
    )


def _build_jina_colbert_v2(runtime: Runtime, spec: SolutionSpec) -> PdfOcrRetrievalSolution:
    return _build_exact_maxsim(
        runtime,
        spec,
        embedding_key="jina-colbert-v2",
        chunker=lambda value: value.text_chunker,
    )


SPECS = (
    SolutionSpec(
        id="paddleocr-v5-mobile-mlateon-exact-maxsim-chunks",
        title="PP-OCRv5 Mobile + mLateOn · exact MaxSim",
        description=(
            "OCR every page, encode overlapping text chunks as token vectors with multilingual mLateOn, "
            "score every chunk with GPU-accelerated exact MaxSim, then collapse chunk hits to PDF documents."
        ),
        tags=("ocr", "late-interaction", "maxsim", "exact", "self-hosted", "chunked", "multilingual"),
        factory=_build_mlateon,
        readme_path="docs/solutions/paddleocr-v5-mobile-mlateon-exact-maxsim.md",
    ),
    SolutionSpec(
        id="paddleocr-v5-mobile-colbert-v2-exact-maxsim-chunks",
        title="PP-OCRv5 Mobile + ColBERTv2 · exact MaxSim",
        description=(
            "OCR every page, create short English text chunks for ColBERTv2's native context, "
            "then score every chunk with GPU-accelerated exact MaxSim."
        ),
        tags=("ocr", "late-interaction", "maxsim", "exact", "self-hosted", "chunked", "english-only"),
        factory=_build_colbert_v2,
        readme_path="docs/solutions/paddleocr-v5-mobile-colbert-v2-exact-maxsim.md",
        languages=("en",),
    ),
    SolutionSpec(
        id="paddleocr-v5-mobile-jina-colbert-v2-exact-maxsim-chunks",
        title="PP-OCRv5 Mobile + Jina-ColBERT-v2 · exact MaxSim",
        description=(
            "OCR every page, encode shared text chunks with multilingual Jina-ColBERT-v2, "
            "then score every chunk with GPU-accelerated exact MaxSim."
        ),
        tags=(
            "ocr",
            "late-interaction",
            "maxsim",
            "exact",
            "self-hosted",
            "chunked",
            "multilingual",
            "non-commercial-license",
        ),
        factory=_build_jina_colbert_v2,
        readme_path="docs/solutions/paddleocr-v5-mobile-jina-colbert-v2-exact-maxsim.md",
    ),
)
