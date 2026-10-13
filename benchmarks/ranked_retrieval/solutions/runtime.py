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
    DashScopeTextEmbedding,
    JinaTextEmbedding,
    OpenRouterEmbedding,
    SentenceTransformersEmbedding,
    VoyageEmbedding,
)
from modern_ir_bench.rerankers import CrossEncoderReranker, OpenRouterReranker, VoyageReranker
from modern_ir_bench.retrieval import HuggingFaceOffsetTokenizer, TextInput, TokenTextChunker
from modern_ir_bench.retrieval.indexes.milvus import MilvusLite, MilvusServer

BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
QWEN3_0_6B_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN3_4B_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
BGE_RERANKER_V2_M3_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
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
    rerank_devices = tuple(
        device.strip() for device in os.environ.get("MIR_RERANK_DEVICES", "cuda:0").split(",") if device.strip()
    )

    def openrouter_embedding(*, model: str, dimension: int, cache_name: str) -> OpenRouterEmbedding:
        return OpenRouterEmbedding(
            model=model,
            dimension=dimension,
            batch_size=int(os.environ.get("MIR_OPENROUTER_BATCH_SIZE", "16")),
            document_input_type="search_document",
            query_input_type="search_query",
            cache_path=project_root / f".runtime/openrouter/{cache_name}.sqlite3",
            request_concurrency=int(os.environ.get("MIR_OPENROUTER_CONCURRENCY", "4")),
        )

    return Runtime(
        target=target,
        cache_root=project_root / ".runtime",
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
            "8b-openrouter": openrouter_embedding(
                model="qwen/qwen3-embedding-8b",
                dimension=4096,
                cache_name="qwen3-embedding-8b",
            ),
        },
        managed={
            "text-embedding-3-large": openrouter_embedding(
                model="openai/text-embedding-3-large",
                dimension=3072,
                cache_name="text-embedding-3-large",
            ),
            "gemini-embedding-2": openrouter_embedding(
                model="google/gemini-embedding-2",
                dimension=3072,
                cache_name="gemini-embedding-2",
            ),
            "pplx-embed-v1-4b": openrouter_embedding(
                model="perplexity/pplx-embed-v1-4b",
                dimension=2560,
                cache_name="pplx-embed-v1-4b",
            ),
            "nemotron-3-embed-1b": openrouter_embedding(
                model="nvidia/nemotron-3-embed-1b:free",
                dimension=2048,
                cache_name="nemotron-3-embed-1b",
            ),
            "mistral-embed-2312": openrouter_embedding(
                model="mistralai/mistral-embed-2312",
                dimension=1024,
                cache_name="mistral-embed-2312",
            ),
            "voyage-4": openrouter_embedding(
                model="voyageai/voyage-4",
                dimension=1024,
                cache_name="voyage-4",
            ),
            "voyage-4-lite": openrouter_embedding(
                model="voyageai/voyage-4-lite",
                dimension=1024,
                cache_name="voyage-4-lite",
            ),
            "text-embedding-3-small": openrouter_embedding(
                model="openai/text-embedding-3-small",
                dimension=1536,
                cache_name="text-embedding-3-small",
            ),
            "pplx-embed-v1-0-6b": openrouter_embedding(
                model="perplexity/pplx-embed-v1-0.6b",
                dimension=1024,
                cache_name="pplx-embed-v1-0-6b",
            ),
            "jina-embeddings-v5-text-small": JinaTextEmbedding(
                cache_path=project_root / ".runtime/jina/jina-embeddings-v5-text-small.sqlite3",
                batch_size=int(os.environ.get("MIR_JINA_BATCH_SIZE", "32")),
                request_concurrency=int(os.environ.get("MIR_JINA_CONCURRENCY", "4")),
                resolve_ip=os.environ.get("MIR_JINA_RESOLVE_IP"),
            ),
            "text-embedding-v4": DashScopeTextEmbedding(
                cache_path=project_root / ".runtime/dashscope/text-embedding-v4.sqlite3",
                batch_size=int(os.environ.get("MIR_DASHSCOPE_BATCH_SIZE", "10")),
                request_concurrency=int(os.environ.get("MIR_DASHSCOPE_CONCURRENCY", "8")),
                query_instruction=QUERY_INSTRUCTION,
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
        rerankers={
            "voyage-rerank-3": VoyageReranker(
                model="rerank-3",
                request_concurrency=int(os.environ.get("MIR_VOYAGE_RERANK_CONCURRENCY", "4")),
                request_interval_seconds=float(os.environ.get("MIR_VOYAGE_RERANK_INTERVAL", "1.0")),
            ),
            "bge-reranker-v2-m3": CrossEncoderReranker(
                model="BAAI/bge-reranker-v2-m3",
                revision=BGE_RERANKER_V2_M3_REVISION,
                max_length=4096,
                batch_size=int(os.environ.get("MIR_BGE_RERANK_BATCH_SIZE", "2")),
                devices=rerank_devices or None,
                use_fp16=True,
            ),
            "cohere-rerank-4-pro": OpenRouterReranker(
                model="cohere/rerank-4-pro",
                request_concurrency=int(os.environ.get("MIR_OPENROUTER_RERANK_CONCURRENCY", "8")),
            ),
        },
    )
