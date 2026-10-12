"""Shared, lazily initialized components for PDF Retrieval Solutions."""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.model_settings import (
    EMBEDDING_GEMMA_2_MODEL,
    EMBEDDING_GEMMA_2_REVISION,
    WEMM_2B_MODEL,
    WEMM_2B_REVISION,
    stable_local_model_kwargs,
)
from benchmarks.pdf_retrieval.solutions.types import Runtime
from modern_ir_bench.embeddings import (
    BgeM3Embedding,
    OpenRouterEmbedding,
    SentenceTransformersEmbedding,
    VoyageMultimodalEmbedding,
)
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget
from modern_ir_bench.solutions import PaddleOcrExtractor, PdfPageRenderer

BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def build_runtime(*, target: MilvusTarget, project_root: Path) -> Runtime:
    runtime_root = project_root / ".runtime"
    device = os.environ.get("MIR_DEVICE")
    return Runtime(
        target=target,
        ocr=PaddleOcrExtractor(
            device=os.environ.get("MIR_OCR_DEVICE", "cpu"),
            cache_dir=runtime_root / "pdf-ocr/pp-ocrv5-mobile",
        ),
        bge_m3=BgeM3Embedding(
            revision=BGE_M3_REVISION,
            batch_size=int(os.environ.get("MIR_BGE_BATCH_SIZE", "12")),
            device=device,
            use_fp16=True,
        ),
        voyage_text=OpenRouterEmbedding(
            model="voyageai/voyage-4-large",
            dimension=1024,
            batch_size=int(os.environ.get("MIR_OPENROUTER_BATCH_SIZE", "32")),
            document_input_type="document",
            query_input_type="query",
            cache_path=runtime_root / "openrouter/voyage-4-large.sqlite3",
            request_concurrency=int(os.environ.get("MIR_OPENROUTER_CONCURRENCY", "4")),
        ),
        voyage_multimodal=VoyageMultimodalEmbedding(
            model="voyage-multimodal-3.5",
            dimension=1024,
            batch_size=int(os.environ.get("MIR_VOYAGE_MULTIMODAL_BATCH_SIZE", "32")),
            cache_path=runtime_root / "pdf-embeddings/voyage-multimodal-3.5.sqlite3",
            request_concurrency=int(os.environ.get("MIR_VOYAGE_MULTIMODAL_CONCURRENCY", "2")),
        ),
        embedding_gemma=SentenceTransformersEmbedding(
            model=EMBEDDING_GEMMA_2_MODEL,
            revision=EMBEDDING_GEMMA_2_REVISION,
            batch_size=int(os.environ.get("MIR_EMBEDDING_GEMMA_IMAGE_BATCH_SIZE", "8")),
            device=device,
            normalize=True,
            query_prompt_name="SearchQuery",
            dimension=768,
            model_kwargs=stable_local_model_kwargs(device),
            config_kwargs={"audio_config": None},
        ),
        wemm=SentenceTransformersEmbedding(
            model=WEMM_2B_MODEL,
            revision=WEMM_2B_REVISION,
            batch_size=int(os.environ.get("MIR_WEMM_IMAGE_BATCH_SIZE", "2")),
            device=device,
            normalize=True,
            trust_remote_code=True,
            dimension=2048,
            model_kwargs=stable_local_model_kwargs(device),
            document_method="encode_document",
            query_method="encode_query",
        ),
        page_renderer=PdfPageRenderer(
            cache_dir=runtime_root / "pdf-pages/pdfium-v1",
            scale=2.0,
            max_side=1600,
            jpeg_quality=88,
        ),
    )
