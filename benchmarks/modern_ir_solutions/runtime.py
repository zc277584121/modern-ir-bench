"""Shared, lazily loaded runtime components for all Solution definitions."""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.modern_ir_solutions.types import Runtime
from modern_ir_bench.embeddings import BgeM3Embedding, SentenceTransformersEmbedding, VoyageEmbedding
from modern_ir_bench.retrieval import HuggingFaceOffsetTokenizer, TextInput, TokenTextChunker
from modern_ir_bench.retrieval.indexes.milvus import MilvusServer

BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
QWEN3_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant documents that answer or satisfy it."


def text_of(value: TextInput) -> str:
    return value.text


def build_runtime() -> Runtime:
    project_root = Path(__file__).resolve().parents[2]
    target = MilvusServer(
        os.environ.get("MIR_MILVUS_URI", "http://127.0.0.1:19530"),
        token=os.environ.get("MIR_MILVUS_TOKEN", ""),
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
        qwen=SentenceTransformersEmbedding(
            model="Qwen/Qwen3-Embedding-4B",
            revision=QWEN3_REVISION,
            batch_size=int(os.environ.get("MIR_QWEN_BATCH_SIZE", "4")),
            device=os.environ.get("MIR_DEVICE"),
            normalize=True,
            query_prompt=f"Instruct: {QUERY_INSTRUCTION}\nQuery: ",
            dimension=2560,
        ),
        bge_m3=BgeM3Embedding(
            revision=BGE_M3_REVISION,
            batch_size=int(os.environ.get("MIR_BGE_BATCH_SIZE", "12")),
            device=os.environ.get("MIR_DEVICE"),
            use_fp16=True,
        ),
    )
