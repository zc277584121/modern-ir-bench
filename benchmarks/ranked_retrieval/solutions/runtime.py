"""Shared, lazily loaded runtime components for all Solution definitions."""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.model_settings import (
    EMBEDDING_GEMMA_2_MODEL,
    EMBEDDING_GEMMA_2_REVISION,
    stable_local_model_kwargs,
)
from benchmarks.ranked_retrieval.solutions.types import Runtime
from modern_ir_bench.embeddings import (
    BgeM3Embedding,
    OpenRouterEmbedding,
    SentenceTransformersEmbedding,
    VoyageEmbedding,
)
from modern_ir_bench.retrieval import HuggingFaceOffsetTokenizer, TextInput, TokenTextChunker
from modern_ir_bench.retrieval.indexes.milvus import MilvusLite, MilvusServer

BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
QWEN3_0_6B_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN3_4B_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant documents that answer or satisfy it."


def text_of(value: TextInput) -> str:
    return value.text


def build_runtime() -> Runtime:
    project_root = Path(__file__).resolve().parents[3]
    device = os.environ.get("MIR_DEVICE")
    lite_path = os.environ.get("MIR_MILVUS_LITE_PATH")
    target = (
        MilvusLite(lite_path)
        if lite_path
        else MilvusServer(
            os.environ.get("MIR_MILVUS_URI", "http://127.0.0.1:19530"),
            token=os.environ.get("MIR_MILVUS_TOKEN", ""),
        )
    )
    return Runtime(
        target=target,
        chunker=TokenTextChunker(HuggingFaceOffsetTokenizer()),
        voyage=VoyageEmbedding(
            model="voyage-4-large",
            batch_size=128,
            max_batch_tokens=100_000,
            cache_path=project_root / ".runtime/voyage-4-large.sqlite3",
        ),
        qwen={
            "0.6b-local": SentenceTransformersEmbedding(
                model="Qwen/Qwen3-Embedding-0.6B",
                revision=QWEN3_0_6B_REVISION,
                batch_size=int(os.environ.get("MIR_QWEN_0_6B_BATCH_SIZE", "24")),
                device=device,
                normalize=True,
                query_prompt=f"Instruct: {QUERY_INSTRUCTION}\nQuery: ",
                dimension=1024,
            ),
            "4b-local": SentenceTransformersEmbedding(
                model="Qwen/Qwen3-Embedding-4B",
                revision=QWEN3_4B_REVISION,
                batch_size=int(os.environ.get("MIR_QWEN_4B_BATCH_SIZE", "4")),
                device=device,
                normalize=True,
                query_prompt=f"Instruct: {QUERY_INSTRUCTION}\nQuery: ",
                dimension=2560,
            ),
            "8b-openrouter": OpenRouterEmbedding(
                model="qwen/qwen3-embedding-8b",
                dimension=4096,
                batch_size=int(os.environ.get("MIR_OPENROUTER_BATCH_SIZE", "64")),
                document_input_type="search_document",
                query_input_type="search_query",
                cache_path=project_root / ".runtime/openrouter/qwen3-embedding-8b.sqlite3",
            ),
        },
        embedding_gemma=SentenceTransformersEmbedding(
            model=EMBEDDING_GEMMA_2_MODEL,
            revision=EMBEDDING_GEMMA_2_REVISION,
            batch_size=int(os.environ.get("MIR_EMBEDDING_GEMMA_BATCH_SIZE", "24")),
            device=device,
            normalize=True,
            document_prompt_name="Document",
            query_prompt_name="SearchQuery",
            dimension=768,
            model_kwargs=stable_local_model_kwargs(device),
            config_kwargs={"vision_config": None, "audio_config": None},
        ),
        bge_m3=BgeM3Embedding(
            revision=BGE_M3_REVISION,
            batch_size=int(os.environ.get("MIR_BGE_BATCH_SIZE", "12")),
            device=os.environ.get("MIR_DEVICE"),
            use_fp16=True,
        ),
    )
