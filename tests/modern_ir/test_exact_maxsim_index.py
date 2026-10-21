from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.pdf_retrieval.solutions.maxsim import SPECS
from modern_ir_bench.retrieval.indexes import TorchExactMaxSimIndex


def _vectors(*rows: tuple[float, float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def test_exact_maxsim_scores_every_document() -> None:
    session = TorchExactMaxSimIndex(device="cpu").open(dimension=2)
    session.add(
        ["mixed", "north", "east"],
        [
            _vectors((1.0, 0.0), (0.0, 1.0)),
            _vectors((1.0, 0.0)),
            _vectors((0.0, 1.0)),
        ],
    )
    session.seal()
    try:
        hits = session.search([_vectors((1.0, 0.0), (0.0, 1.0))], top_k=3)[0]
    finally:
        session.close()

    assert [hit.id for hit in hits] == ["mixed", "east", "north"]
    assert [hit.score for hit in hits] == pytest.approx([2.0, 1.0, 1.0])


def test_exact_maxsim_normalizes_token_vectors() -> None:
    session = TorchExactMaxSimIndex(device="cpu").open(dimension=2)
    session.add(["scaled", "diagonal"], [_vectors((10.0, 0.0)), _vectors((1.0, 1.0))])
    session.seal()
    try:
        hits = session.search([_vectors((3.0, 0.0))], top_k=2)[0]
        metadata = session.metadata
    finally:
        session.close()

    assert [hit.id for hit in hits] == ["scaled", "diagonal"]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 2**-0.5])
    assert metadata == {
        "backend": "torch-exact-maxsim",
        "device": "cpu",
        "dtype": "float32",
        "dimension": 2,
        "count": 2,
        "token_vectors": 2,
        "normalize": True,
        "exact": True,
    }


def test_exact_maxsim_rejects_duplicate_ids() -> None:
    session = TorchExactMaxSimIndex(device="cpu").open(dimension=2)
    session.add(["document"], [_vectors((1.0, 0.0))])

    with pytest.raises(ValueError, match="Duplicate ids"):
        session.add(["document"], [_vectors((0.0, 1.0))])

    session.close()


@pytest.mark.parametrize(
    ("solution_id", "embedding_key", "chunker_name"),
    [
        ("paddleocr-v5-mobile-mlateon-exact-maxsim-chunks", "mlateon", "standard"),
        ("paddleocr-v5-mobile-colbert-v2-exact-maxsim-chunks", "colbert-v2", "short"),
        ("paddleocr-v5-mobile-jina-colbert-v2-exact-maxsim-chunks", "jina-colbert-v2", "standard"),
    ],
)
def test_pdf_late_interaction_solutions_use_exact_maxsim(
    solution_id: str,
    embedding_key: str,
    chunker_name: str,
) -> None:
    spec = next(item for item in SPECS if item.id == solution_id)
    embeddings = {
        key: SimpleNamespace(batch_size=8, device="cpu")
        for key in ("mlateon", "colbert-v2", "jina-colbert-v2")
    }
    standard_chunker = object()
    short_chunker = object()
    runtime = SimpleNamespace(
        ocr=object(),
        late_interaction=embeddings,
        text_chunker=standard_chunker,
        short_text_chunker=short_chunker,
    )

    solution = spec.build(runtime)

    assert isinstance(solution.text_solution.index, TorchExactMaxSimIndex)
    assert solution.text_solution.index.device == "cpu"
    expected_chunker = standard_chunker if chunker_name == "standard" else short_chunker
    assert solution.text_solution.chunker is expected_chunker
    assert solution.text_solution.embedding is embeddings[embedding_key]
