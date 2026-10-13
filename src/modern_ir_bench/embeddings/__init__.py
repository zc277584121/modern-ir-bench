"""Embedding component contracts and implementations."""

from modern_ir_bench.embeddings.bge_m3 import BgeM3Embedding
from modern_ir_bench.embeddings.learned_sparse import (
    GteMultilingualSparseEmbedding,
    SentenceTransformersSparseEmbedding,
)
from modern_ir_bench.embeddings.managed_text import DashScopeTextEmbedding, JinaTextEmbedding
from modern_ir_bench.embeddings.openrouter import OpenRouterEmbedding
from modern_ir_bench.embeddings.ovis_omni import OvisOmniEmbedding
from modern_ir_bench.embeddings.protocols import DenseEmbedding
from modern_ir_bench.embeddings.sentence_transformers import (
    SentenceTransformersEmbedding,
)
from modern_ir_bench.embeddings.sparse import SparseEmbedding, SparseVector
from modern_ir_bench.embeddings.voyage import VoyageEmbedding
from modern_ir_bench.embeddings.voyage_multimodal import VoyageMultimodalEmbedding

__all__ = [
    "BgeM3Embedding",
    "DashScopeTextEmbedding",
    "DenseEmbedding",
    "GteMultilingualSparseEmbedding",
    "JinaTextEmbedding",
    "OpenRouterEmbedding",
    "OvisOmniEmbedding",
    "SentenceTransformersEmbedding",
    "SentenceTransformersSparseEmbedding",
    "SparseEmbedding",
    "SparseVector",
    "VoyageEmbedding",
    "VoyageMultimodalEmbedding",
]
