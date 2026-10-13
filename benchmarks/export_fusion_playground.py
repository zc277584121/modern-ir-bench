"""Export reusable atomic Top-K evidence for the static fusion playground."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.pdf_retrieval.definition import DATASET_REVISION as PDF_DATASET_REVISION
from benchmarks.pdf_retrieval.definition import RELEASE_ID as PDF_RELEASE_ID
from benchmarks.pdf_retrieval.definition import build_task as build_pdf_task
from benchmarks.ranked_retrieval.definition import DATASET_REVISION as TEXT_DATASET_REVISION
from benchmarks.ranked_retrieval.definition import RELEASE_ID as TEXT_RELEASE_ID
from benchmarks.ranked_retrieval.definition import build_task as build_text_task
from modern_ir_bench.exporters.fusion_playground import (
    PlaygroundDimension,
    PlaygroundRoute,
    build_playground_benchmark,
    write_playground_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / ".runtime/retrieval-candidates.sqlite3"
DEFAULT_OUTPUT = PROJECT_ROOT / "space/data/fusion-playground.json.gz"
TEXT_NAMESPACE = f"ranked:{TEXT_DATASET_REVISION}:sparse-fusion:v1"
PDF_TEXT_NAMESPACE = f"pdf:{PDF_DATASET_REVISION}:ocr-bm25-bge-m3-full:v1"
PDF_VISUAL_NAMESPACE = f"pdf:{PDF_DATASET_REVISION}:ocr-bm25-voyage-multimodal-pages:v1"

QUERY_DIMENSIONS = (
    PlaygroundDimension("language", "Language", {"en": "English", "zh": "Chinese"}),
    PlaygroundDimension("query_intent", "Intent"),
    PlaygroundDimension("expression", "Expression"),
    PlaygroundDimension("constraint_level", "Constraint level"),
)


def build_payload(cache_path: Path) -> list[dict[str, object]]:
    """Build the two currently published playground datasets."""

    text = build_playground_benchmark(
        task=build_text_task(),
        dataset_id=TEXT_RELEASE_ID,
        cache_path=cache_path,
        dimensions=QUERY_DIMENSIONS,
        routes=(
            PlaygroundRoute(
                "bge-m3-dense",
                "BGE-M3 dense",
                "Dense",
                TEXT_NAMESPACE,
                "internal-bge-m3-dense-chunk",
            ),
            PlaygroundRoute(
                "qwen3-embedding-8b",
                "Qwen3 Embedding 8B",
                "Dense",
                TEXT_NAMESPACE,
                "internal-qwen3-embedding-8b-chunk",
            ),
            PlaygroundRoute(
                "bge-m3-sparse",
                "BGE-M3 sparse",
                "Learned sparse",
                TEXT_NAMESPACE,
                "internal-bge-m3-sparse-chunk",
            ),
            PlaygroundRoute(
                "gte-multilingual-sparse",
                "GTE Multilingual sparse",
                "Learned sparse",
                TEXT_NAMESPACE,
                "internal-gte-multilingual-base-sparse-chunk",
            ),
            PlaygroundRoute(
                "bm25",
                "BM25",
                "Lexical",
                TEXT_NAMESPACE,
                "internal-bm25-chunk",
                "lower",
            ),
        ),
    )
    pdf = build_playground_benchmark(
        task=build_pdf_task(),
        dataset_id=PDF_RELEASE_ID,
        cache_path=cache_path,
        dimensions=QUERY_DIMENSIONS,
        routes=(
            PlaygroundRoute(
                "ocr-bge-m3",
                "OCR + BGE-M3",
                "Dense",
                PDF_TEXT_NAMESPACE,
                "internal-bge-m3-full",
            ),
            PlaygroundRoute(
                "voyage-multimodal-3-5",
                "Voyage Multimodal 3.5",
                "Visual",
                PDF_VISUAL_NAMESPACE,
                "internal-voyage-multimodal-pages",
            ),
            PlaygroundRoute(
                "ocr-bm25",
                "OCR + BM25",
                "Lexical",
                PDF_TEXT_NAMESPACE,
                "internal-bm25-full",
                "lower",
            ),
        ),
    )
    return [text, pdf]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    write_playground_payload(build_payload(arguments.cache), arguments.output)


if __name__ == "__main__":
    main()
