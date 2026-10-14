from __future__ import annotations

import pytest

from modern_ir_bench.retrieval import BM25LanguageConfig, MilvusBM25Solution, RetrievalResource, TextInput
from modern_ir_bench.retrieval.indexes.milvus import MilvusLite


def test_milvus_bm25_supports_benchmark_language_analyzers(tmp_path) -> None:
    cases = (
        ("en", ("orchard apple harvest", "marine weather report"), "apple harvest"),
        ("zh", ("果园苹果采收记录", "海洋天气报告"), "苹果采收"),
    )
    for language, documents, query in cases:
        solution = MilvusBM25Solution(
            id=f"bm25-{language}",
            title="BM25",
            target=MilvusLite(tmp_path / f"{language}.db"),
        )
        session = solution.prepare(
            [RetrievalResource(id=f"d{index}", value=TextInput(text, language)) for index, text in enumerate(documents)]
        )
        try:
            results = session.search_batch([TextInput(query, language)], top_k=2)
            assert results[0][0].id == "d0"
            assert len(results[0]) == 2
        finally:
            session.close()


def test_milvus_bm25_exposes_language_configuration(tmp_path) -> None:
    analyzer = {"tokenizer": "standard", "filter": ["lowercase"]}
    config = BM25LanguageConfig(analyzer=analyzer, k1=1.8, b=0.6)
    solution = MilvusBM25Solution(
        id="bm25-configured",
        title="BM25 configured",
        target=MilvusLite(tmp_path / "configured.db"),
        language_configs={"en": config},
    )
    session = solution.prepare([RetrievalResource(id="d0", value=TextInput("orchard apple harvest", "en"))])
    try:
        assert session.metadata["analyzer"] == analyzer
        assert session.metadata["k1"] == 1.8
        assert session.metadata["b"] == 0.6
    finally:
        session.close()


@pytest.mark.parametrize(
    ("arguments", "message"),
    (({"k1": 0}, "k1"), ({"b": 1.1}, "b")),
)
def test_bm25_language_config_rejects_invalid_scoring_parameters(arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        BM25LanguageConfig(analyzer={"type": "english"}, **arguments)
