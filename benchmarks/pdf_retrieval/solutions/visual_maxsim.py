"""OCR-free visual late-interaction PDF Retrieval Solutions."""

from __future__ import annotations

from functools import partial

from benchmarks.pdf_retrieval.solutions.types import Runtime, SolutionSpec
from modern_ir_bench.retrieval import ChunkedMultiVectorRetrievalSolution, PdfInput, TextInput
from modern_ir_bench.retrieval.indexes import TorchExactMaxSimIndex
from modern_ir_bench.solutions import PdfPageRenderer


def _query_text(value: TextInput) -> str:
    return value.text


def _render_page_paths(renderer: PdfPageRenderer, value: PdfInput) -> list[str]:
    return [str(path) for path in renderer.render(value)]


def _build_visual_exact_maxsim(
    runtime: Runtime,
    spec: SolutionSpec,
    *,
    embedding_key: str,
) -> ChunkedMultiVectorRetrievalSolution:
    embedding = runtime.visual_late_interaction[embedding_key]
    return ChunkedMultiVectorRetrievalSolution(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        embedding=embedding,
        index=TorchExactMaxSimIndex(
            device=embedding.device,
            dtype="float16",
            block_vector_limit=1_000_000,
        ),
        chunker=partial(_render_page_paths, runtime.page_renderer),
        chunk_batch_size=embedding.batch_size,
        candidate_multiplier=3,
        query_input=_query_text,
    )


def _build_colsmol_256m(runtime: Runtime, spec: SolutionSpec) -> ChunkedMultiVectorRetrievalSolution:
    return _build_visual_exact_maxsim(runtime, spec, embedding_key="colsmol-256m")


def _build_colsmol_500m(runtime: Runtime, spec: SolutionSpec) -> ChunkedMultiVectorRetrievalSolution:
    return _build_visual_exact_maxsim(runtime, spec, embedding_key="colsmol-500m")


def _build_colpali_v1_3(runtime: Runtime, spec: SolutionSpec) -> ChunkedMultiVectorRetrievalSolution:
    return _build_visual_exact_maxsim(runtime, spec, embedding_key="colpali-v1.3")


def _build_colqwen2_v1_0(runtime: Runtime, spec: SolutionSpec) -> ChunkedMultiVectorRetrievalSolution:
    return _build_visual_exact_maxsim(runtime, spec, embedding_key="colqwen2-v1.0")


def _build_colqwen_2_5(runtime: Runtime, spec: SolutionSpec) -> ChunkedMultiVectorRetrievalSolution:
    return _build_visual_exact_maxsim(runtime, spec, embedding_key="colqwen2.5")


SPECS = (
    SolutionSpec(
        id="colsmol-500m-page-exact-maxsim",
        title="ColSmol 500M · page exact MaxSim",
        description=(
            "Encode PDF page images directly with ColSmol 500M and score every page with exact MaxSim, "
            "without OCR or approximate candidate generation."
        ),
        tags=("visual", "late-interaction", "maxsim", "exact", "self-hosted", "page-level", "ocr-free", "english-only"),
        factory=_build_colsmol_500m,
        readme_path="docs/solutions/colsmol-500m-page-exact-maxsim.md",
        languages=("en",),
    ),
    SolutionSpec(
        id="colpali-v1-3-page-exact-maxsim",
        title="ColPali v1.3 · page exact MaxSim",
        description=(
            "Encode PDF page images directly with ColPali v1.3 and score every page with exact MaxSim, "
            "without OCR or approximate candidate generation."
        ),
        tags=("visual", "late-interaction", "maxsim", "exact", "self-hosted", "page-level", "ocr-free", "english-only"),
        factory=_build_colpali_v1_3,
        readme_path="docs/solutions/colpali-v1-3-page-exact-maxsim.md",
        languages=("en",),
    ),
    SolutionSpec(
        id="colqwen2-v1-0-page-exact-maxsim",
        title="ColQwen2 v1.0 · page exact MaxSim",
        description=(
            "Encode PDF page images directly with ColQwen2 v1.0 and score every page with exact MaxSim, "
            "without OCR or approximate candidate generation."
        ),
        tags=("visual", "late-interaction", "maxsim", "exact", "self-hosted", "page-level", "ocr-free"),
        factory=_build_colqwen2_v1_0,
        readme_path="docs/solutions/colqwen2-v1-0-page-exact-maxsim.md",
        languages=("en", "zh"),
    ),
    SolutionSpec(
        id="colsmol-256m-page-exact-maxsim",
        title="ColSmol 256M · page exact MaxSim",
        description=(
            "Encode PDF page images directly with ColSmol 256M and score every page with exact MaxSim, "
            "without OCR or approximate candidate generation."
        ),
        tags=("visual", "late-interaction", "maxsim", "exact", "self-hosted", "page-level", "ocr-free", "english-only"),
        factory=_build_colsmol_256m,
        readme_path="docs/solutions/colsmol-256m-page-exact-maxsim.md",
        languages=("en",),
    ),
    SolutionSpec(
        id="colqwen2-5-page-exact-maxsim",
        title="ColQwen2.5 · page exact MaxSim",
        description=(
            "Encode PDF page images directly with ColQwen2.5 and score every page with exact MaxSim, "
            "without OCR or approximate candidate generation."
        ),
        tags=("visual", "late-interaction", "maxsim", "exact", "self-hosted", "page-level", "ocr-free"),
        factory=_build_colqwen_2_5,
        readme_path="docs/solutions/colqwen2-5-page-exact-maxsim.md",
        languages=("en", "zh"),
    ),
)
