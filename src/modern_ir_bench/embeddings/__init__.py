"""Embedding component contracts and implementations."""

from modern_ir_bench.embeddings.bge_m3 import BgeM3Embedding
from modern_ir_bench.embeddings.protocols import DenseEmbedding
from modern_ir_bench.embeddings.sentence_transformers import (
    SentenceTransformersEmbedding,
)
from modern_ir_bench.embeddings.sparse import SparseEmbedding, SparseVector
from modern_ir_bench.embeddings.voyage import VoyageEmbedding

__all__ = [
    "BgeM3Embedding",
    "DenseEmbedding",
    "SentenceTransformersEmbedding",
    "SparseEmbedding",
    "SparseVector",
    "VoyageEmbedding",
]
