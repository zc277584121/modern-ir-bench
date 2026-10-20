"""Dense index backends and independent verification."""

from modern_ir_bench.retrieval.indexes.flat import NumpyFlatIndex
from modern_ir_bench.retrieval.indexes.milvus import (
    MilvusDenseIndex,
    MilvusLite,
    MilvusMultiVectorIndex,
    MilvusServer,
    MilvusSparseIndex,
    ZillizCloud,
)
from modern_ir_bench.retrieval.indexes.protocols import (
    DenseIndex,
    DenseIndexSession,
    MultiVectorIndex,
    MultiVectorIndexSession,
)
from modern_ir_bench.retrieval.indexes.sparse import SparseIndex, SparseIndexSession
from modern_ir_bench.retrieval.indexes.verified import (
    DenseIndexAudit,
    VerifiedDenseIndex,
)

__all__ = [
    "DenseIndex",
    "DenseIndexAudit",
    "DenseIndexSession",
    "MilvusDenseIndex",
    "MilvusLite",
    "MilvusMultiVectorIndex",
    "MilvusServer",
    "MilvusSparseIndex",
    "MultiVectorIndex",
    "MultiVectorIndexSession",
    "NumpyFlatIndex",
    "SparseIndex",
    "SparseIndexSession",
    "VerifiedDenseIndex",
    "ZillizCloud",
]
