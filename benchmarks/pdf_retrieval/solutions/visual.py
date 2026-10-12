"""OCR-free visual PDF Retrieval Solutions."""

from __future__ import annotations

from functools import partial

from benchmarks.pdf_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution, PdfInput, TextInput
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex
from modern_ir_bench.solutions import PdfPageRenderer


def _query_text(value: TextInput) -> str:
    return value.text


def _render_page_images(renderer: PdfPageRenderer, value: PdfInput) -> list[dict[str, str]]:
    return [{"image": str(path)} for path in renderer.render(value)]


def _build_managed_visual_pages(
    runtime: Runtime,
    spec: SolutionSpec,
    *,
    embedding_key: str,
) -> ChunkedDenseRetrievalSolution:
    embedding = runtime.managed_visual[embedding_key]
    return ChunkedDenseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding=embedding,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix=f"modern_ir_pdf_{spec.id.replace('-', '_')}",
        ),
        chunker=runtime.page_renderer.render,
        chunk_batch_size=embedding.batch_size * embedding.request_concurrency,
        candidate_multiplier=3,
        query_input=_query_text,
    )


def _build_gemini_pages(runtime: Runtime, spec: SolutionSpec) -> ChunkedDenseRetrievalSolution:
    return _build_managed_visual_pages(runtime, spec, embedding_key="gemini-embedding-2")


def _build_nemotron_pages(runtime: Runtime, spec: SolutionSpec) -> ChunkedDenseRetrievalSolution:
    return _build_managed_visual_pages(runtime, spec, embedding_key="llama-nemotron-embed-vl-1b-v2")


def _build_ovis_pages(runtime: Runtime, spec: SolutionSpec) -> ChunkedDenseRetrievalSolution:
    return ChunkedDenseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding=runtime.ovis_omni,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_pdf_ovis_omni_3b_pages",
        ),
        chunker=runtime.page_renderer.render,
        chunk_batch_size=32,
        candidate_multiplier=3,
        query_input=_query_text,
    )


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


def _build_embedding_gemma_pages(runtime: Runtime, spec: SolutionSpec) -> ChunkedDenseRetrievalSolution:
    return ChunkedDenseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding=runtime.embedding_gemma,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_pdf_embedding_gemma_2_pages",
        ),
        chunker=partial(_render_page_images, runtime.page_renderer),
        chunk_batch_size=runtime.embedding_gemma.batch_size,
        candidate_multiplier=3,
        query_input=_query_text,
    )


def _build_wemm_pages(runtime: Runtime, spec: SolutionSpec) -> ChunkedDenseRetrievalSolution:
    return ChunkedDenseRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding=runtime.wemm,
        index=MilvusDenseIndex(
            target=runtime.target,
            metric="COSINE",
            index_type="FLAT",
            collection_prefix="modern_ir_pdf_wemm_2b_pages",
        ),
        chunker=partial(_render_page_images, runtime.page_renderer),
        chunk_batch_size=64,
        candidate_multiplier=3,
        query_input=_query_text,
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
    SolutionSpec(
        id="embedding-gemma-2-pages",
        title="EmbeddingGemma 2 · page images",
        description="Render PDF pages as images, retrieve them with local EmbeddingGemma 2, then collapse to documents.",
        tags=("visual", "dense", "self-hosted", "page-level", "ocr-free"),
        factory=_build_embedding_gemma_pages,
        readme_path="docs/solutions/embedding-gemma-2-pages.md",
    ),
    SolutionSpec(
        id="wemm-embedding-2b-pages",
        title="WeMM Embedding 2B · page images",
        description="Render PDF pages as images, retrieve them with local WeMM Embedding 2B, then collapse to documents.",
        tags=("visual", "dense", "self-hosted", "page-level", "ocr-free"),
        factory=_build_wemm_pages,
        readme_path="docs/solutions/wemm-embedding-2b-pages.md",
    ),
    SolutionSpec(
        id="gemini-embedding-2-pages",
        title="Gemini Embedding 2 · page images",
        description="Render PDF pages as images, retrieve them with Gemini Embedding 2, then collapse to documents.",
        tags=("visual", "dense", "managed-api", "page-level", "ocr-free", "multimodal"),
        factory=_build_gemini_pages,
        readme_path="docs/solutions/gemini-embedding-2-pages.md",
    ),
    SolutionSpec(
        id="llama-nemotron-embed-vl-1b-v2-pages",
        title="Llama Nemotron Embed VL 1B V2 · page images",
        description="Render PDF pages as images, retrieve them with Llama Nemotron Embed VL 1B V2, then collapse to documents.",
        tags=("visual", "dense", "managed-api", "page-level", "ocr-free", "multimodal"),
        factory=_build_nemotron_pages,
        readme_path="docs/solutions/llama-nemotron-embed-vl-1b-v2-pages.md",
    ),
    SolutionSpec(
        id="ovis-omni-embedding-3b-pages",
        title="Ovis Omni Embedding 3B · page images",
        description="Render PDF pages as images, retrieve them with local Ovis Omni Embedding 3B, then collapse to documents.",
        tags=("visual", "dense", "self-hosted", "page-level", "ocr-free", "multimodal"),
        factory=_build_ovis_pages,
        readme_path="docs/solutions/ovis-omni-embedding-3b-pages.md",
    ),
)
