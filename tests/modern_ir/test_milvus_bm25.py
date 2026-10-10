from __future__ import annotations

from modern_ir_bench.retrieval import MilvusBM25Solution, RetrievalResource, TextInput
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
