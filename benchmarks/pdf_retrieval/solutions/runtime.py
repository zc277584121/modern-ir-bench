"""Shared, lazily initialized components for PDF Retrieval Solutions."""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.model_settings import (
    COLBERT_V2_MODEL,
    COLBERT_V2_REVISION,
    COLQWEN_2_5_MODEL,
    COLQWEN_2_5_REVISION,
    COLSMOL_256M_MODEL,
    COLSMOL_256M_REVISION,
    EMBEDDING_GEMMA_2_MODEL,
    EMBEDDING_GEMMA_2_REVISION,
    JINA_COLBERT_V2_MODEL,
    JINA_COLBERT_V2_REVISION,
    JINA_XLM_ROBERTA_CODE_REVISION,
    MLATEON_MODEL,
    MLATEON_REVISION,
    OVIS_OMNI_3B_MODEL,
    OVIS_OMNI_3B_REVISION,
    WEMM_2B_MODEL,
    WEMM_2B_REVISION,
    stable_local_model_kwargs,
)
from benchmarks.pdf_retrieval.solutions.types import Runtime
from modern_ir_bench.embeddings import (
    BgeM3Embedding,
    OpenRouterEmbedding,
    OvisOmniEmbedding,
    SentenceTransformersEmbedding,
    SentenceTransformersMultiVectorEmbedding,
    VoyageEmbedding,
    VoyageMultimodalEmbedding,
)
from modern_ir_bench.rerankers import VoyageReranker
from modern_ir_bench.retrieval import ChunkingPolicy, HuggingFaceOffsetTokenizer, TokenTextChunker
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget
from modern_ir_bench.solutions import PaddleOcrExtractor, PdfPageRenderer

BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def build_runtime(*, target: MilvusTarget, project_root: Path) -> Runtime:
    runtime_root = project_root / ".runtime"
    device = os.environ.get("MIR_DEVICE")
    chunk_tokenizer = HuggingFaceOffsetTokenizer()
    return Runtime(
        target=target,
        cache_root=runtime_root,
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
        voyage_text=VoyageEmbedding(
            model="voyage-4-large",
            dimension=1024,
            batch_size=int(os.environ.get("MIR_VOYAGE_BATCH_SIZE", "128")),
            max_batch_tokens=int(os.environ.get("MIR_VOYAGE_MAX_BATCH_TOKENS", "100000")),
            cache_path=runtime_root / "voyage-4-large.sqlite3",
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
        late_interaction={
            "mlateon": SentenceTransformersMultiVectorEmbedding(
                model=MLATEON_MODEL,
                revision=MLATEON_REVISION,
                dimension=128,
                batch_size=int(os.environ.get("MIR_MLATEON_BATCH_SIZE", "16")),
                device=device,
                normalize=True,
                cache_path=runtime_root / "multi-vector/mlateon.sqlite3",
                model_kwargs=stable_local_model_kwargs(device),
            ),
            "colbert-v2": SentenceTransformersMultiVectorEmbedding(
                model=COLBERT_V2_MODEL,
                revision=COLBERT_V2_REVISION,
                dimension=128,
                batch_size=int(os.environ.get("MIR_COLBERT_V2_BATCH_SIZE", "64")),
                device=device,
                normalize=True,
                cache_path=runtime_root / "multi-vector/colbert-v2.sqlite3",
                model_kwargs=stable_local_model_kwargs(device),
            ),
            "jina-colbert-v2": SentenceTransformersMultiVectorEmbedding(
                model=JINA_COLBERT_V2_MODEL,
                revision=JINA_COLBERT_V2_REVISION,
                dimension=128,
                batch_size=int(os.environ.get("MIR_JINA_COLBERT_V2_BATCH_SIZE", "1")),
                device=device,
                normalize=True,
                trust_remote_code=True,
                cache_path=runtime_root / "multi-vector/jina-colbert-v2-doc4096.sqlite3",
                document_length=4096,
                model_kwargs={
                    **stable_local_model_kwargs(device),
                    "code_revision": JINA_XLM_ROBERTA_CODE_REVISION,
                },
            ),
        },
        visual_late_interaction={
            "colsmol-256m": SentenceTransformersMultiVectorEmbedding(
                model=COLSMOL_256M_MODEL,
                revision=COLSMOL_256M_REVISION,
                dimension=128,
                batch_size=int(os.environ.get("MIR_COLSMOL_256M_BATCH_SIZE", "4")),
                device=device,
                normalize=True,
                cache_path=runtime_root / "multi-vector/colsmol-256m-pages.sqlite3",
                model_kwargs=stable_local_model_kwargs(device),
            ),
            "colqwen2.5": SentenceTransformersMultiVectorEmbedding(
                model=COLQWEN_2_5_MODEL,
                revision=COLQWEN_2_5_REVISION,
                dimension=128,
                batch_size=int(os.environ.get("MIR_COLQWEN_2_5_BATCH_SIZE", "1")),
                device=device,
                normalize=True,
                cache_path=runtime_root / "multi-vector/colqwen2.5-v0.2-pages.sqlite3",
                model_kwargs=stable_local_model_kwargs(device),
            ),
        },
        wemm=SentenceTransformersEmbedding(
            model=WEMM_2B_MODEL,
            revision=WEMM_2B_REVISION,
            batch_size=int(os.environ.get("MIR_WEMM_IMAGE_BATCH_SIZE", "4")),
            device=device,
            normalize=True,
            trust_remote_code=True,
            dimension=2048,
            model_kwargs=stable_local_model_kwargs(device),
            document_method="encode_document",
            query_method="encode_query",
        ),
        managed_visual={
            "gemini-embedding-2": OpenRouterEmbedding(
                model="google/gemini-embedding-2",
                dimension=3072,
                batch_size=int(os.environ.get("MIR_OPENROUTER_IMAGE_BATCH_SIZE", "8")),
                document_input_type="search_document",
                query_input_type="search_query",
                cache_path=runtime_root / "openrouter/gemini-embedding-2-images.sqlite3",
                request_concurrency=int(os.environ.get("MIR_OPENROUTER_IMAGE_CONCURRENCY", "4")),
            ),
            "llama-nemotron-embed-vl-1b-v2": OpenRouterEmbedding(
                model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
                dimension=2048,
                batch_size=int(os.environ.get("MIR_OPENROUTER_IMAGE_BATCH_SIZE", "8")),
                document_input_type="search_document",
                query_input_type="search_query",
                cache_path=runtime_root / "openrouter/llama-nemotron-embed-vl-1b-v2-images.sqlite3",
                request_concurrency=int(os.environ.get("MIR_OPENROUTER_IMAGE_CONCURRENCY", "4")),
            ),
        },
        ovis_omni=OvisOmniEmbedding(
            model=OVIS_OMNI_3B_MODEL,
            revision=OVIS_OMNI_3B_REVISION,
            cache_path=runtime_root / "pdf-embeddings/ovis-omni-embedding-3b.sqlite3",
            batch_size=int(os.environ.get("MIR_OVIS_BATCH_SIZE", "4")),
            device=device,
            max_image_pixels=int(os.environ.get("MIR_OVIS_MAX_IMAGE_PIXELS", "2007040")),
        ),
        page_renderer=PdfPageRenderer(
            cache_dir=runtime_root / "pdf-pages/pdfium-v1",
            scale=2.0,
            max_side=1600,
            jpeg_quality=88,
        ),
        text_chunker=TokenTextChunker(chunk_tokenizer),
        short_text_chunker=TokenTextChunker(
            chunk_tokenizer,
            ChunkingPolicy(
                target_tokens=128,
                min_tokens=64,
                max_tokens=160,
                overlap_tokens=32,
                paragraph_search_tokens=32,
            ),
        ),
        reranker=VoyageReranker(
            model="rerank-3",
            request_concurrency=int(os.environ.get("MIR_VOYAGE_RERANK_CONCURRENCY", "4")),
            request_interval_seconds=float(os.environ.get("MIR_VOYAGE_RERANK_INTERVAL", "1.0")),
        ),
    )
