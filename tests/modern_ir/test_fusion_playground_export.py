from __future__ import annotations

import gzip
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from modern_ir_bench.core import MetricSet
from modern_ir_bench.exporters.fusion_playground import (
    PlaygroundDimension,
    PlaygroundRoute,
    build_playground_benchmark,
    write_playground_payload,
)
from modern_ir_bench.metrics import NDCG
from modern_ir_bench.retrieval import RetrievalResource, SearchHit, TextInput
from modern_ir_bench.retrieval.candidate_cache import SqliteCandidateCache
from modern_ir_bench.tasks import RankedRetrieval


def _task() -> RankedRetrieval:
    return RankedRetrieval(
        id="task",
        title="Task",
        description="Test task.",
        version="1",
        datasets={
            "dataset": DatasetDict(
                documents=Dataset.from_list(
                    [
                        {"document_id": "a", "content": "A", "language": "en"},
                        {"document_id": "b", "content": "B", "language": "en"},
                    ]
                ),
                queries=Dataset.from_list(
                    [
                        {
                            "query_id": "q",
                            "query": "question",
                            "language": "en",
                        }
                    ]
                ),
                qrels=Dataset.from_list(
                    [{"query_id": "q", "document_id": "b", "relevance": 1}]
                ),
            )
        },
        dataset_versions={"dataset": "revision"},
        metrics=MetricSet(primary=NDCG(k=10)),
        query_value=lambda row: TextInput(text=row["query"], language=row["language"]),
        pool_field="language",
    )


def test_build_playground_benchmark_reads_atomic_cache(tmp_path: Path) -> None:
    task = _task()
    cache_path = tmp_path / "candidates.sqlite3"
    resources = [RetrievalResource(id="a", value=None), RetrievalResource(id="b", value=None)]
    query = TextInput(text="question", language="en")
    cache = SqliteCandidateCache(cache_path, namespace="namespace")
    cache.write(
        corpus_key=cache.corpus_key(resources),
        component_id="component",
        queries=[query],
        depth=10,
        rankings=[[SearchHit(id="b", score=2.0), SearchHit(id="a", score=1.0)]],
    )

    payload = build_playground_benchmark(
        task=task,
        dataset_id="dataset",
        cache_path=cache_path,
        routes=(PlaygroundRoute("route", "Route", "Dense", "namespace", "component"),),
        dimensions=(PlaygroundDimension("language", "Language", {"en": "English"}),),
        depth=10,
    )

    assert payload["documents"] == ["a", "b"]
    assert payload["queries"] == [{"id": "q", "labels": ["en"], "relevant": [1]}]
    assert payload["routes"][0]["rankings"] == [[1, 2.0, 0, 1.0]]


def test_write_playground_payload_is_valid_gzip(tmp_path: Path) -> None:
    output = tmp_path / "playground.json.gz"
    write_playground_payload([{"task_id": "task"}], output)

    with gzip.open(output, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)

    assert payload == {"schema_version": 1, "benchmarks": [{"task_id": "task"}]}
