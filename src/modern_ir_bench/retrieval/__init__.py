"""Reusable retrieval components that remain internal to Solutions."""

from modern_ir_bench.retrieval.bm25 import MilvusBM25Solution
from modern_ir_bench.retrieval.chunked import ChunkedDenseRetrievalSolution
from modern_ir_bench.retrieval.chunking import ChunkingPolicy, HuggingFaceOffsetTokenizer, TokenTextChunker
from modern_ir_bench.retrieval.dense import DenseRetrievalSolution
from modern_ir_bench.retrieval.sparse import ChunkedSparseRetrievalSolution
from modern_ir_bench.retrieval.types import (
    MappedResourceSource,
    RetrievalResource,
    SearchHit,
    SearchSession,
    TextInput,
)

__all__ = [
    "ChunkedDenseRetrievalSolution",
    "ChunkedSparseRetrievalSolution",
    "ChunkingPolicy",
    "DenseRetrievalSolution",
    "HuggingFaceOffsetTokenizer",
    "MappedResourceSource",
    "MilvusBM25Solution",
    "RetrievalResource",
    "SearchHit",
    "SearchSession",
    "TextInput",
    "TokenTextChunker",
]
