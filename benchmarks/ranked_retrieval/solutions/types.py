"""Shared values for the executable Modern IR Solution catalog."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from modern_ir_bench.core import Solution
from modern_ir_bench.embeddings import BgeM3Embedding, DenseEmbedding, VoyageEmbedding
from modern_ir_bench.retrieval import TokenTextChunker
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget


@dataclass(frozen=True)
class Runtime:
    target: MilvusTarget
    chunker: TokenTextChunker
    voyage: VoyageEmbedding
    qwen: dict[str, DenseEmbedding]
    embedding_gemma: DenseEmbedding
    bge_m3: BgeM3Embedding


@dataclass(frozen=True)
class SolutionSpec:
    id: str
    title: str
    description: str
    tags: tuple[str, ...]
    factory: Callable[[Runtime, SolutionSpec], Solution]
    readme_path: str

    def build(self, runtime: Runtime) -> Solution:
        return self.factory(runtime, self)
