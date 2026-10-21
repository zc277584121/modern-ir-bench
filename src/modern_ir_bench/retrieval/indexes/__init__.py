"""Dense index backends and independent verification."""

from modern_ir_bench.retrieval.indexes.exact_maxsim import TorchExactMaxSimIndex
from modern_ir_bench.retrieval.indexes.flat import NumpyFlatIndex
from modern_ir_bench.retrieval.indexes.milvus import (
    MilvusDenseIndex,
    MilvusLite,
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
    "MilvusServer",
    "MilvusSparseIndex",
    "MultiVectorIndex",
    "MultiVectorIndexSession",
    "NumpyFlatIndex",
    "SparseIndex",
    "SparseIndexSession",
    "TorchExactMaxSimIndex",
    "VerifiedDenseIndex",
    "ZillizCloud",
]
