"""OCR-free visual PDF Retrieval Solutions."""

from __future__ import annotations

from benchmarks.pdf_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution, TextInput
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex


def _query_text(value: TextInput) -> str:
    return value.text


def build_voyage_pages_component(
    runtime: Runtime,
    *,
    solution_id: str,
    title: str,
    description: str,
) -> ChunkedDenseRetrievalSolution:
    """Build the reusable PDF-page visual retrieval path."""
    return ChunkedDenseRetrievalSolution(
        id=solution_id,
        title=title,
        description=description,
        embedding=runtime.voyage_multimodal,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_pdf_voyage_multimodal_pages",
        ),
        chunker=runtime.page_renderer.render,
        chunk_batch_size=(
            runtime.voyage_multimodal.batch_size * runtime.voyage_multimodal.request_concurrency
        ),
        candidate_multiplier=3,
        query_input=_query_text,
    )


def _build_voyage_pages(runtime: Runtime, spec: SolutionSpec) -> ChunkedDenseRetrievalSolution:
    return build_voyage_pages_component(
        runtime,
        solution_id=spec.id,
        title=spec.title,
        description=spec.description,
    )


SPECS = (
    SolutionSpec(
        id="voyage-multimodal-3-5-pages",
        title="Voyage Multimodal 3.5 · page images",
        description="Render PDF pages as images, retrieve pages without OCR, then collapse page hits to documents.",
        tags=("visual", "dense", "managed-api", "page-level", "ocr-free"),
        factory=_build_voyage_pages,
        readme_path="docs/solutions/voyage-multimodal-3-5-pages.md",
    ),
)
