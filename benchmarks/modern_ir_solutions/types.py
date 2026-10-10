"""Shared values for the executable Modern IR Solution catalog."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from modern_ir_bench.core import Solution
from modern_ir_bench.embeddings import BgeM3Embedding, SentenceTransformersEmbedding, VoyageEmbedding
from modern_ir_bench.retrieval import TokenTextChunker
from modern_ir_bench.retrieval.indexes.milvus import MilvusServer


@dataclass(frozen=True)
class SolutionSpec:
    id: str
    title: str
    description: str
    tags: tuple[str, ...]
    builder: Callable[[Runtime], Solution]
    readme_path: str


@dataclass(frozen=True)
class Runtime:
    target: MilvusServer
    chunker: TokenTextChunker
    voyage: VoyageEmbedding
    qwen: SentenceTransformersEmbedding
    bge_m3: BgeM3Embedding
