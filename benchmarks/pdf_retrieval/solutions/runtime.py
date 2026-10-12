"""Shared, lazily initialized components for PDF Retrieval Solutions."""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.pdf_retrieval.solutions.types import Runtime
from modern_ir_bench.embeddings import BgeM3Embedding, VoyageEmbedding, VoyageMultimodalEmbedding
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget
from modern_ir_bench.solutions import PaddleOcrExtractor, PdfPageRenderer

BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def build_runtime(*, target: MilvusTarget, project_root: Path) -> Runtime:
    runtime_root = project_root / ".runtime"
    return Runtime(
        target=target,
        ocr=PaddleOcrExtractor(
            device=os.environ.get("MIR_OCR_DEVICE", "cpu"),
            cache_dir=runtime_root / "pdf-ocr/pp-ocrv5-mobile",
        ),
        bge_m3=BgeM3Embedding(
            revision=BGE_M3_REVISION,
            batch_size=int(os.environ.get("MIR_BGE_BATCH_SIZE", "12")),
            device=os.environ.get("MIR_DEVICE"),
            use_fp16=True,
        ),
        voyage_text=VoyageEmbedding(
            model="voyage-4-large",
            dimension=1024,
            batch_size=32,
            max_batch_tokens=9_000,
            cache_path=runtime_root / "pdf-embeddings/voyage-4-large.sqlite3",
        ),
        voyage_multimodal=VoyageMultimodalEmbedding(
            model="voyage-multimodal-3.5",
            dimension=1024,
            batch_size=int(os.environ.get("MIR_VOYAGE_MULTIMODAL_BATCH_SIZE", "32")),
            cache_path=runtime_root / "pdf-embeddings/voyage-multimodal-3.5.sqlite3",
            request_concurrency=int(os.environ.get("MIR_VOYAGE_MULTIMODAL_CONCURRENCY", "2")),
        ),
        page_renderer=PdfPageRenderer(
            cache_dir=runtime_root / "pdf-pages/pdfium-v1",
            scale=2.0,
            max_side=1600,
            jpeg_quality=88,
        ),
    )
