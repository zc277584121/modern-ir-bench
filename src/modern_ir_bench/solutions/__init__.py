"""Reference and demonstration solutions."""

from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution, DenseRetrievalSolution
from modern_ir_bench.solutions.replay import (
    SavedRankingSolution,
    build_saved_ranking_solutions,
    load_release_replay_solutions,
)
from modern_ir_bench.solutions.text import (
    BM25Solution,
    CharacterNGramSolution,
    HybridSolution,
)

__all__ = [
    "BM25Solution",
    "CharacterNGramSolution",
    "ChunkedDenseRetrievalSolution",
    "DenseRetrievalSolution",
    "HybridSolution",
    "SavedRankingSolution",
    "build_saved_ranking_solutions",
    "load_release_replay_solutions",
]
