"""Reference and demonstration solutions."""

from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution, DenseRetrievalSolution
from modern_ir_bench.solutions.text import (
    CharacterNGramSolution,
    HybridSolution,
    InMemoryBM25Solution,
)

__all__ = [
    "CharacterNGramSolution",
    "ChunkedDenseRetrievalSolution",
    "DenseRetrievalSolution",
    "HybridSolution",
    "InMemoryBM25Solution",
]
