"""Runtime and catalog values for PDF Retrieval Solutions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from modern_ir_bench.core import Solution
from modern_ir_bench.embeddings import (
    BgeM3Embedding,
    OpenRouterEmbedding,
    OvisOmniEmbedding,
    SentenceTransformersEmbedding,
    VoyageEmbedding,
    VoyageMultimodalEmbedding,
)
from modern_ir_bench.rerankers import Reranker
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget
from modern_ir_bench.solutions import PaddleOcrExtractor, PdfPageRenderer


@dataclass(frozen=True)
class Runtime:
    target: MilvusTarget
    cache_root: Path
    ocr: PaddleOcrExtractor
    bge_m3: BgeM3Embedding
    voyage_text: VoyageEmbedding
    voyage_multimodal: VoyageMultimodalEmbedding
    embedding_gemma: SentenceTransformersEmbedding
    wemm: SentenceTransformersEmbedding
    managed_visual: dict[str, OpenRouterEmbedding]
    ovis_omni: OvisOmniEmbedding
    page_renderer: PdfPageRenderer
    reranker: Reranker


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
