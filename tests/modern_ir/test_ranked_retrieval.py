from __future__ import annotations

from datasets import Dataset, DatasetDict, Features, Value

from modern_ir_bench import MetricSet, RunProvenance
from modern_ir_bench.metrics import NDCG, AveragePrecision, Recall
from modern_ir_bench.solutions import InMemoryBM25Solution
from modern_ir_bench.tasks import RankedRetrieval


def dataset() -> DatasetDict:
    return DatasetDict(
        documents=Dataset.from_dict(
            {
                "document_id": ["d1", "d2", "d3"],
                "content": ["alpha beta", "gamma delta", "alpha gamma"],
            },
            features=RankedRetrieval.dataset_features["documents"],
        ),
        queries=Dataset.from_dict(
            {
                "query_id": ["q1", "q2"],
                "query": ["alpha beta", "gamma delta"],
            },
            features=RankedRetrieval.dataset_features["queries"],
        ),
        qrels=Dataset.from_dict(
            {
                "query_id": ["q1", "q1", "q2", "q2"],
                "document_id": ["d1", "d3", "d2", "d1"],
                "relevance": [1, 0, 1, 0],
            },
            features=RankedRetrieval.dataset_features["qrels"],
        ),
    )


def provenance() -> RunProvenance:
    return RunProvenance(
        source_commit="0123456789abcdef",
        source_module="tests.modern_ir.test_ranked_retrieval",
        source_path="tests/modern_ir/test_ranked_retrieval.py",
        source_line=1,
        repository_url="https://github.com/example/modern-ir-bench",
        created_at="2026-09-09T00:00:00+00:00",
    )


def test_ranked_retrieval_uses_three_table_hf_dataset_contract() -> None:
    task = RankedRetrieval(
        id="ranked-retrieval-test",
        title="Ranked Retrieval Test",
        description="Test task",
        version="1",
        datasets={"tiny": dataset()},
        dataset_versions={"tiny": "1"},
        metrics=MetricSet(
            primary=NDCG(k=2),
            secondary=[AveragePrecision(k=2), Recall(k=2)],
        ),
        top_k=2,
        query_batch_size=1,
    )
    report = task.run(
        [InMemoryBM25Solution(id="bm25", title="BM25")],
        provenance=provenance(),
        status="test",
    )

    assert {record["value"] for record in report.records} == {1.0}
    observation = report.observations["ranked-retrieval-test__tiny__bm25"]
    assert observation.column_names == [
        "query_id",
        "expected_ids",
        "expected_relevance",
        "ranked_ids",
        "scores",
    ]
    assert observation[0]["expected_ids"] == ["d1"]
    assert observation[1]["expected_ids"] == ["d2"]


def test_ranked_retrieval_rejects_qrels_that_cross_candidate_pools() -> None:
    data = dataset()
    data["documents"] = data["documents"].add_column("language", ["en", "zh", "en"])
    data["queries"] = data["queries"].add_column("language", ["en", "zh"])
    task = RankedRetrieval(
        id="pooled-retrieval-test",
        title="Pooled Retrieval Test",
        description="Test task",
        version="1",
        datasets={"tiny": data},
        dataset_versions={"tiny": "1"},
        metrics=MetricSet(primary=NDCG(k=2)),
        pool_field="language",
    )

    try:
        task.validate_dataset(data)
    except ValueError as error:
        assert "crosses independent pools" in str(error)
    else:
        raise AssertionError("cross-pool Qrels must be rejected")


def test_ranked_retrieval_searches_independent_candidate_pools() -> None:
    data = DatasetDict(
        documents=Dataset.from_dict(
            {
                "document_id": ["en-relevant", "en-other", "zh-relevant", "zh-other"],
                "content": ["shared apple", "shared ocean", "shared apple", "shared ocean"],
                "language": ["en", "en", "zh", "zh"],
            },
            features=Features(
                {
                    **RankedRetrieval.dataset_features["documents"],
                    "language": Value("string"),
                }
            ),
        ),
        queries=Dataset.from_dict(
            {
                "query_id": ["q-en", "q-zh"],
                "query": ["apple", "apple"],
                "language": ["en", "zh"],
            },
            features=Features(
                {
                    **RankedRetrieval.dataset_features["queries"],
                    "language": Value("string"),
                }
            ),
        ),
        qrels=Dataset.from_dict(
            {
                "query_id": ["q-en", "q-zh"],
                "document_id": ["en-relevant", "zh-relevant"],
                "relevance": [1, 1],
            },
            features=RankedRetrieval.dataset_features["qrels"],
        ),
    )
    task = RankedRetrieval(
        id="pooled-retrieval-test",
        title="Pooled Retrieval Test",
        description="Test task",
        version="1",
        datasets={"tiny": data},
        dataset_versions={"tiny": "1"},
        metrics=MetricSet(primary=NDCG(k=1)),
        top_k=1,
        pool_field="language",
    )

    report = task.run(
        [InMemoryBM25Solution(id="bm25", title="BM25")],
        provenance=provenance(),
        status="test",
    )

    observations = report.observations["pooled-retrieval-test__tiny__bm25"]
    assert observations["ranked_ids"] == [["en-relevant"], ["zh-relevant"]]


def test_saved_rankings_cannot_cross_candidate_pools() -> None:
    data = DatasetDict(
        documents=Dataset.from_dict(
            {
                "document_id": ["en-document", "zh-document"],
                "content": ["English", "中文"],
                "language": ["en", "zh"],
            }
        ),
        queries=Dataset.from_dict(
            {
                "query_id": ["en-query", "zh-query"],
                "query": ["English", "中文"],
                "language": ["en", "zh"],
            }
        ),
        qrels=Dataset.from_dict(
            {
                "query_id": ["en-query", "zh-query"],
                "document_id": ["en-document", "zh-document"],
                "relevance": [1, 1],
            },
            features=RankedRetrieval.dataset_features["qrels"],
        ),
    )
    task = RankedRetrieval(
        id="pooled-retrieval-test",
        title="Pooled Retrieval Test",
        description="Test task",
        version="1",
        datasets={"tiny": data},
        dataset_versions={"tiny": "1"},
        metrics=MetricSet(primary=NDCG(k=1)),
        top_k=1,
        pool_field="language",
    )

    try:
        task.observations_from_rankings(
            data,
            {
                "en-query": ["zh-document"],
                "zh-query": ["en-document"],
            },
        )
    except ValueError as error:
        assert "crosses independent pools" in str(error)
    else:
        raise AssertionError("cross-pool saved rankings must be rejected")
