"""Embedding component contracts and implementations."""

from modern_ir_bench.embeddings.bge_m3 import BgeM3Embedding
from modern_ir_bench.embeddings.cached_multi_vector import CachedMultiVectorEmbedding
from modern_ir_bench.embeddings.jina_v4 import JinaV4MultiVectorEmbedding
from modern_ir_bench.embeddings.learned_sparse import (
    GteMultilingualSparseEmbedding,
    OnnxSpladeEmbedding,
    SentenceTransformersSparseEmbedding,
)
from modern_ir_bench.embeddings.managed_text import DashScopeTextEmbedding, JinaTextEmbedding
from modern_ir_bench.embeddings.multi_vector import SentenceTransformersMultiVectorEmbedding
from modern_ir_bench.embeddings.openrouter import OpenRouterEmbedding
from modern_ir_bench.embeddings.ovis_omni import OvisOmniEmbedding
from modern_ir_bench.embeddings.protocols import DenseEmbedding, MultiVectorEmbedding, MultiVectors
from modern_ir_bench.embeddings.sentence_transformers import (
    SentenceTransformersEmbedding,
)
from modern_ir_bench.embeddings.sparse import SparseEmbedding, SparseVector
from modern_ir_bench.embeddings.voyage import VoyageEmbedding
from modern_ir_bench.embeddings.voyage_multimodal import VoyageMultimodalEmbedding

__all__ = [
    "BgeM3Embedding",
    "CachedMultiVectorEmbedding",
    "DashScopeTextEmbedding",
    "DenseEmbedding",
    "GteMultilingualSparseEmbedding",
    "JinaTextEmbedding",
    "JinaV4MultiVectorEmbedding",
    "MultiVectorEmbedding",
    "MultiVectors",
    "OnnxSpladeEmbedding",
    "OpenRouterEmbedding",
    "OvisOmniEmbedding",
    "SentenceTransformersEmbedding",
    "SentenceTransformersMultiVectorEmbedding",
    "SentenceTransformersSparseEmbedding",
    "SparseEmbedding",
    "SparseVector",
    "VoyageEmbedding",
    "VoyageMultimodalEmbedding",
]
