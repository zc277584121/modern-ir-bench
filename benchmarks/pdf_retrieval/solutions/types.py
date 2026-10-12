"""Runtime and catalog values for PDF Retrieval Solutions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from modern_ir_bench.core import Solution
from modern_ir_bench.embeddings import (
    BgeM3Embedding,
    OpenRouterEmbedding,
    SentenceTransformersEmbedding,
    VoyageMultimodalEmbedding,
)
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget
from modern_ir_bench.solutions import PaddleOcrExtractor, PdfPageRenderer


@dataclass(frozen=True)
class Runtime:
    target: MilvusTarget
    ocr: PaddleOcrExtractor
    bge_m3: BgeM3Embedding
    voyage_text: OpenRouterEmbedding
    voyage_multimodal: VoyageMultimodalEmbedding
    embedding_gemma: SentenceTransformersEmbedding
    wemm: SentenceTransformersEmbedding
    page_renderer: PdfPageRenderer


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
