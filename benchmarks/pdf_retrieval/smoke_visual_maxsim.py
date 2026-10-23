"""Exercise real visual encoders, independent scoring, and model-free cache replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np

from benchmarks.pdf_retrieval.solutions import build_solutions
from benchmarks.pdf_retrieval.solutions.visual_maxsim import SPECS
from modern_ir_bench.embeddings import JinaV4MultiVectorEmbedding
from modern_ir_bench.retrieval import PdfInput, RetrievalResource, TextInput
from modern_ir_bench.retrieval.indexes.milvus import MilvusLite

CASES = {
    "en": (
        (
            "fruit",
            "FRUIT SHOP\n\nApples and bananas for sale.\nFresh apples are red and green.\nBananas are yellow fruit.",
            "Where can I buy apples and bananas?",
        ),
        (
            "space",
            "SOLAR SYSTEM\n\nPlanets orbit the Sun.\nMars is the red planet.\nJupiter is the largest planet.",
            "Which planet is the largest?",
        ),
    ),
    "zh": (
        (
            "fruit",
            "水果商店\n\n本店出售新鲜苹果和香蕉。\n苹果有红色和绿色。\n香蕉是黄色的水果。",
            "哪里可以买到苹果和香蕉？",
        ),
        (
            "space",
            "太阳系\n\n行星围绕太阳运行。\n火星是一颗红色的行星。\n木星是太阳系中最大的行星。",
            "太阳系中哪颗行星最大？",
        ),
    ),
}
JINA_SCORE_SOURCE = (
    "https://huggingface.co/jinaai/jina-embeddings-v4/blob/"
    "d347c952d22e0aabbeabe2a28c205c6c3d5dfaad/modeling_colqwen_duo.py"
)


def _resources(root: Path, language: str, font_path: Path | None) -> list[RetrievalResource[PdfInput]]:
    from PIL import Image, ImageDraw, ImageFont

    # CJK fallback fonts can omit Latin glyphs; keep English on Pillow's Latin font.
    font = ImageFont.truetype(str(font_path), 28) if language == "zh" and font_path else ImageFont.load_default(size=28)
    resources = []
    for name, text, _ in CASES[language]:
        path = root / f"{language}-{name}.pdf"
        with Image.new("RGB", (640, 480), "white") as image:
            ImageDraw.Draw(image).multiline_text((32, 48), text, font=font, fill="black", spacing=12)
            image.save(path, "PDF", resolution=96)
        resources.append(RetrievalResource(id=f"{language}-{name}", value=PdfInput(path=str(path), language=language)))
    return resources


def _official_scores(embedding, queries, documents):
    import torch

    # Match the index's explicit float32 normalization, including older caches
    # whose native BF16 outputs only approximate unit length.
    qs = [torch.nn.functional.normalize(torch.from_numpy(value), dim=1) for value in queries]
    ps = [torch.nn.functional.normalize(torch.from_numpy(value), dim=1) for value in documents]
    if isinstance(embedding, JinaV4MultiVectorEmbedding):
        # Jina's current native API omits scoring. This is its published padded
        # einsum/max/sum scorer at JINA_SCORE_SOURCE, evaluated in float32.
        q = torch.nn.utils.rnn.pad_sequence(qs, batch_first=True, padding_value=0)
        p = torch.nn.utils.rnn.pad_sequence(ps, batch_first=True, padding_value=0)
        return torch.einsum("bnd,csd->bcns", q, p).max(dim=3).values.sum(dim=2).numpy(), JINA_SCORE_SOURCE
    from sentence_transformers.util import maxsim

    return maxsim(qs, ps).numpy(), "sentence_transformers.util.maxsim (MultiVectorEncoder.similarity)"


def run_smoke(solution_id: str, root: Path, *, font_path: Path | None = None) -> dict:
    import torch

    spec = next(spec for spec in SPECS if spec.id == solution_id)
    if root.resolve() == Path(__file__).resolve().parents[2]:
        raise ValueError("Smoke work directory must be separate from the benchmark project root")
    root.mkdir(parents=True, exist_ok=True)
    if "zh" in spec.languages and font_path is None:
        font_path = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
        if not font_path.is_file():
            raise ValueError("Bilingual smoke requires --font pointing to a Chinese-capable font")
    target = MilvusLite(root / ".runtime/milvus.db")
    (solution,) = build_solutions(target=target, project_root=root, solution_ids=[solution_id])
    embedding = solution.embedding
    if embedding.cache.path.exists():
        raise ValueError("Cold smoke requires an empty embedding cache; choose a fresh --work-dir")
    device = torch.device(embedding.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    resources = {lang: _resources(root, lang, font_path) for lang in spec.languages}
    queries = {lang: [TextInput(text=query, language=lang) for _, _, query in CASES[lang]] for lang in spec.languages}
    started = perf_counter()
    preparation = solution.pre_encode(
        [resource for group in resources.values() for resource in group],
        [query for group in queries.values() for query in group],
    )
    assert embedding._model is None and embedding.cache_only
    result = {
        "solution": solution.id,
        "model": embedding.model_id,
        "revision": embedding.revision,
        "configuration": embedding.configuration,
        "cache_fingerprint": embedding.cache_fingerprint,
        "device": str(device),
        "batch_size": embedding.batch_size,
        "preparation": preparation,
        "languages": {},
        "query_shapes": [],
        "document_shapes": [],
    }
    if device.type == "cuda":
        result["encoding_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["encoding_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        result["allocated_bytes_after_encoder_release"] = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    scoring_started = perf_counter()
    try:
        # A cache miss must fail, including after constructing a fresh encoder.
        for replay in range(2):
            if replay:
                (solution,) = build_solutions(target=target, project_root=root, solution_ids=[solution_id])
                embedding = solution.embedding
                embedding.cache_only = True
            with patch.object(embedding, "_load_model", side_effect=AssertionError("Cache replay loaded a model")):
                for language in spec.languages:
                    group = resources[language]
                    session = solution.prepare(group)
                    try:
                        rankings = session.search_batch(queries[language], top_k=len(group))
                        serialized = [[asdict(hit) for hit in hits] for hits in rankings]
                        if replay:
                            assert serialized == result["languages"][language]["rankings"]
                            continue
                        pages = [page for resource in group for page in solution.chunker(resource.value)]
                        documents = embedding.encode_documents(pages)
                        query_vectors = embedding.encode_queries([query.text for query in queries[language]])
                        norm_error = 0.0
                        for matrix in [*documents, *query_vectors]:
                            assert matrix.ndim == 2 and matrix.shape[0] > 0 and matrix.shape[1] == embedding.dimension
                            assert np.isfinite(matrix).all()
                            norm_error = max(norm_error, float(np.abs(np.linalg.norm(matrix, axis=1) - 1).max()))
                        assert norm_error < 0.01
                        reference, source = _official_scores(embedding, query_vectors, documents)
                        normalized = [value / np.linalg.norm(value, axis=1, keepdims=True) for value in documents]
                        numpy_scores = np.asarray(
                            [
                                [
                                    (q / np.linalg.norm(q, axis=1, keepdims=True) @ p.T).max(axis=1).sum()
                                    for p in normalized
                                ]
                                for q in query_vectors
                            ]
                        )
                        scores = np.asarray(
                            [
                                [next(hit.score for hit in hits if hit.id == doc.id) for doc in group]
                                for hits in rankings
                            ]
                        )
                        np.testing.assert_allclose(scores, reference, atol=0.02, rtol=0)
                        np.testing.assert_allclose(scores, numpy_scores, atol=0.02, rtol=0)
                        assert [hits[0].id for hits in rankings] == [doc.id for doc in group]
                        result["languages"][language] = {
                            "rankings": serialized,
                            "official_scores": reference.tolist(),
                            "score_reference": source,
                            "numpy_scores": numpy_scores.tolist(),
                            "official_max_absolute_error": float(np.abs(scores - reference).max()),
                            "numpy_max_absolute_error": float(np.abs(scores - numpy_scores).max()),
                            "max_norm_error": norm_error,
                            "metadata": dict(session.metadata),
                            "document_float32_bytes": sum(matrix.nbytes for matrix in documents),
                            "document_index_float16_bytes": sum(matrix.size * 2 for matrix in documents),
                        }
                        result["query_shapes"].extend(list(matrix.shape) for matrix in query_vectors)
                        result["document_shapes"].extend(list(matrix.shape) for matrix in documents)
                    finally:
                        session.close()
    finally:
        solution.release()
    result["scoring_and_replay_seconds"] = perf_counter() - scoring_started
    result["elapsed_seconds"] = perf_counter() - started
    result["embedding_cache_bytes"] = embedding.cache.path.stat().st_size
    result["cache_replay_without_model"] = True
    if device.type == "cuda":
        result["scoring_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["scoring_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", required=True, choices=[spec.id for spec in SPECS])
    parser.add_argument("--output", type=Path, help="Write the smoke report as JSON.")
    parser.add_argument("--work-dir", type=Path, help="Retain isolated smoke inputs and caches here.")
    parser.add_argument("--font", type=Path, help="TrueType/OpenType font for Chinese pages.")
    arguments = parser.parse_args()
    if arguments.work_dir is not None:
        result = run_smoke(arguments.solution, arguments.work_dir, font_path=arguments.font)
    else:
        with TemporaryDirectory(prefix="visual-maxsim-smoke-") as temporary:
            result = run_smoke(arguments.solution, Path(temporary), font_path=arguments.font)
    report = json.dumps(result, indent=2, default=str) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
