"""Smoke-test a real visual MaxSim Solution on two synthetic single-page PDFs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from benchmarks.pdf_retrieval.solutions import build_solutions
from benchmarks.pdf_retrieval.solutions.visual_maxsim import SPECS
from modern_ir_bench.retrieval import PdfInput, RetrievalResource, TextInput
from modern_ir_bench.retrieval.indexes.milvus import MilvusLite


def _resources(root: Path) -> list[RetrievalResource[PdfInput]]:
    from PIL import Image, ImageDraw, ImageFont

    pages = {
        "fruit": "FRUIT SHOP\n\nApples and bananas for sale.\nFresh apples are red and green.\nBananas are yellow fruit.",
        "space": "SOLAR SYSTEM\n\nPlanets orbit the Sun.\nMars is the red planet.\nJupiter is the largest planet.",
    }
    resources = []
    for name, text in pages.items():
        path = root / f"{name}.pdf"
        with Image.new("RGB", (640, 480), "white") as image:
            ImageDraw.Draw(image).multiline_text(
                (32, 48),
                text,
                font=ImageFont.load_default(size=28),
                fill="black",
                spacing=12,
            )
            image.save(path, "PDF", resolution=96)
        resources.append(RetrievalResource(id=name, value=PdfInput(path=str(path), language="en")))
    return resources


def run_smoke(solution_id: str, root: Path) -> dict:
    import torch

    (solution,) = build_solutions(
        target=MilvusLite(root / "unused.db"),
        project_root=root,
        solution_ids=[solution_id],
    )
    embedding = solution.embedding
    device = torch.device(embedding.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    resources = _resources(root)
    queries = [TextInput("Where can I buy apples and bananas?"), TextInput("Which planet is the largest?")]
    session = None
    try:
        session = solution.prepare(resources)
        rankings = session.search_batch(queries, top_k=2)
        pages = [page for resource in resources for page in solution.chunker(resource.value)]
        documents = embedding.encode_documents(pages)
        query_vectors = embedding.encode_queries([query.text for query in queries])
        for matrix in [*documents, *query_vectors]:
            assert matrix.ndim == 2 and matrix.shape[0] > 0 and matrix.shape[1] == 128
            assert np.isfinite(matrix).all()
            np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, atol=0.01)

        reference = []
        for query, hits in zip(query_vectors, rankings, strict=True):
            query = query / np.linalg.norm(query, axis=1, keepdims=True)
            scores = [
                float((query @ (document / np.linalg.norm(document, axis=1, keepdims=True)).T).max(axis=1).sum())
                for document in documents
            ]
            expected = sorted(
                zip([resource.id for resource in resources], scores, strict=True),
                key=lambda item: (-item[1], item[0]),
            )
            assert [hit.id for hit in hits] == [item[0] for item in expected]
            np.testing.assert_allclose([hit.score for hit in hits], [item[1] for item in expected], atol=0.02)
            reference.append(scores)
        assert [hits[0].id for hits in rankings] == ["fruit", "space"]

        result = {
            "solution": solution.id,
            "model": embedding.model_id,
            "revision": embedding.revision,
            "device": str(device),
            "batch_size": embedding.batch_size,
            "query_shapes": [list(matrix.shape) for matrix in query_vectors],
            "document_shapes": [list(matrix.shape) for matrix in documents],
            "rankings": [[asdict(hit) for hit in hits] for hits in rankings],
            "numpy_maxsim_scores": reference,
            "metadata": dict(session.metadata),
            "embedding_cache_bytes": embedding.cache.path.stat().st_size,
            "document_float32_bytes": sum(matrix.nbytes for matrix in documents),
            "document_index_float16_bytes": sum(matrix.size * 2 for matrix in documents),
        }
        if device.type == "cuda":
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        session.close()
        session = None
        solution.release()
        session = solution.prepare(resources)
        assert session.search_batch(queries, top_k=2) == rankings
        assert embedding._model is None
        result["cache_replay_without_model"] = True
        return result
    finally:
        if session is not None:
            session.close()
        solution.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", required=True, choices=[spec.id for spec in SPECS])
    parser.add_argument("--output", type=Path, help="Write the smoke report as JSON.")
    arguments = parser.parse_args()
    with TemporaryDirectory(prefix="visual-maxsim-smoke-") as temporary:
        result = run_smoke(arguments.solution, Path(temporary))
    report = json.dumps(result, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
