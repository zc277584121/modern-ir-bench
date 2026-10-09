"""Reusable retrieval components that remain internal to Solutions."""

from modern_ir_bench.retrieval.chunked import ChunkedDenseRetrievalSolution
from modern_ir_bench.retrieval.chunking import (
    CanonicalTextChunker,
    ChunkingPolicy,
    HuggingFaceOffsetTokenizer,
)
from modern_ir_bench.retrieval.dense import DenseRetrievalSolution
from modern_ir_bench.retrieval.types import (
    MappedResourceSource,
    RetrievalResource,
    SearchHit,
    SearchSession,
)

__all__ = [
    "CanonicalTextChunker",
    "ChunkedDenseRetrievalSolution",
    "ChunkingPolicy",
    "DenseRetrievalSolution",
    "HuggingFaceOffsetTokenizer",
    "MappedResourceSource",
    "RetrievalResource",
    "SearchHit",
    "SearchSession",
]
