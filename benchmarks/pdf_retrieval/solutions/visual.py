"""OCR-free visual PDF Retrieval Solutions."""

from __future__ import annotations

from benchmarks.pdf_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution, TextInput
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex


def _query_text(value: TextInput) -> str:
    return value.text


def _build_voyage_pages(runtime: Runtime, spec: SolutionSpec) -> ChunkedDenseRetrievalSolution:
    return ChunkedDenseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding=runtime.voyage_multimodal,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_pdf_voyage_multimodal_pages",
        ),
        chunker=runtime.page_renderer.render,
        chunk_batch_size=32,
        candidate_multiplier=3,
        query_input=_query_text,
    )


SPECS = (
    SolutionSpec(
        id="voyage-multimodal-3-5-pages",
        title="Voyage Multimodal 3.5 · page images",
        description="Render PDF pages as images, retrieve pages without OCR, then collapse page hits to documents.",
        tags=("visual", "dense", "hosted", "page-level", "ocr-free", "milvus"),
        factory=_build_voyage_pages,
        readme_path="docs/solutions/voyage-multimodal-3-5-pages.md",
    ),
)
