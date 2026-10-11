from __future__ import annotations

from datasets import Dataset, IterableDataset

from modern_ir_bench import RunProvenance
from modern_ir_bench.core.result import RunReport
from modern_ir_bench.exporters import build_space_payload
from modern_ir_bench.metrics import NDCG


def provenance() -> RunProvenance:
    return RunProvenance(
        source_commit="0123456789abcdef",
        source_module="benchmarks.example",
        source_path="benchmarks/example.py",
        source_line=1,
        repository_url="https://github.com/example/modern-ir-bench",
        created_at="2026-09-09T00:00:00+00:00",
    )


def report() -> RunReport:
    output = RunReport()
    output.add_evaluation(
        task_id="retrieval",
        task_title="Retrieval",
        task_description="Find relevant documents.",
        task_version="1.0.0",
        dataset_id="sample",
        dataset_version="1",
        solution_id="bm25",
        solution_title="BM25",
        solution_description="Lexical baseline.",
        primary_metric_id="ndcg@10",
        metrics=[NDCG(k=10)(Dataset.from_list([{"expected_ids": ["a"], "ranked_ids": ["a"]}]))],
        observations=Dataset.from_list([{"query_id": "q", "expected_ids": ["a"], "ranked_ids": ["a"]}]),
        provenance=provenance(),
        status="test",
    )
    return output


def test_space_exporter_is_separate_from_the_run_report() -> None:
    payload = build_space_payload(
        report(),
        release="test-release",
        counts={"documents": 1, "queries": 1},
        languages=("English",),
        breakdowns=(
            {
                "id": "language",
                "label": "Language",
                "values": [{"id": "en", "label": "English", "count": 1, "results": []}],
            },
        ),
    )

    assert payload["benchmark"]["release"] == "test-release"
    assert payload["benchmark"]["counts"] == {"documents": 1, "queries": 1}
    assert payload["benchmark"]["languages"] == ["English"]
    assert payload["breakdowns"][0]["id"] == "language"
    assert "notice" not in payload["benchmark"]
    assert len(payload["tasks"]) == 1
    assert len(payload["solutions"]) == 1
    assert len(payload["results"]) == 1


def test_hugging_face_batches_iterable_dataset_without_a_custom_runtime() -> None:
    stream = IterableDataset.from_generator(lambda: ({"value": value} for value in range(5)))

    batches = list(stream.iter(batch_size=2))

    assert [batch["value"] for batch in batches] == [[0, 1], [2, 3], [4]]
