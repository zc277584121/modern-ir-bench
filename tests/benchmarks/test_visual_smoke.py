from __future__ import annotations

import numpy as np

from benchmarks.pdf_retrieval.smoke_visual_maxsim import _official_scores, _resources
from modern_ir_bench.embeddings import JinaV4MultiVectorEmbedding
from modern_ir_bench.retrieval.indexes.exact_maxsim import TorchExactMaxSimIndex


def test_official_score_reference_matches_index_normalization(tmp_path):
    encoder = JinaV4MultiVectorEmbedding(model="test", revision="fixed", cache_dir=tmp_path)
    query = np.zeros((2, 128), dtype=np.float32)
    document = np.zeros((2, 128), dtype=np.float32)
    query[0, 0], query[1, 1] = 2, 3
    document[0, 0], document[1, 1] = 5, 7
    reference, _ = _official_scores(encoder, [query], [document])
    session = TorchExactMaxSimIndex(device="cpu").open(dimension=128)
    try:
        session.add(["document"], [document])
        session.seal()
        score = session.search([query], top_k=1)[0][0].score
        assert score == reference[0, 0] == 2.0
    finally:
        session.close()
    assert encoder._model is None
    assert not encoder.cache.path.exists()


def test_english_smoke_does_not_use_the_chinese_fallback_font(tmp_path):
    resources = _resources(tmp_path, "en", tmp_path / "chinese-only.ttf")
    assert [resource.id for resource in resources] == ["en-fruit", "en-space"]
    assert all((tmp_path / f"{resource.id}.pdf").is_file() for resource in resources)
