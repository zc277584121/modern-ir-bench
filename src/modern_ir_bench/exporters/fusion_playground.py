"""Export cached atomic rankings for client-side fusion experiments."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from modern_ir_bench.retrieval import RetrievalResource
from modern_ir_bench.retrieval.candidate_cache import SqliteCandidateCache


@dataclass(frozen=True)
class PlaygroundDimension:
    """One query field that users may use to slice live metrics."""

    id: str
    label: str
    value_labels: Mapping[str, str] | None = None


@dataclass(frozen=True)
class PlaygroundRoute:
    """One cached atomic retrieval route exposed to the browser."""

    id: str
    label: str
    kind: str
    namespace: str
    component_id: str
    score_direction: Literal["higher", "lower"] = "higher"


def build_playground_benchmark(
    *,
    task: Any,
    dataset_id: str,
    cache_path: Path,
    routes: Sequence[PlaygroundRoute],
    dimensions: Sequence[PlaygroundDimension],
    depth: int = 50,
) -> dict[str, Any]:
    """Build one compact benchmark payload without running retrieval again."""

    if depth < 10:
        raise ValueError("Playground rankings must contain at least Top-10")
    dataset = task.datasets[dataset_id]
    document_ids = [str(value) for value in dataset["documents"]["document_id"]]
    document_index = {document_id: index for index, document_id in enumerate(document_ids)}
    if len(document_index) != len(document_ids):
        raise ValueError("Document ids must be unique")

    relevant_by_query: dict[str, list[int]] = defaultdict(list)
    for row in dataset["qrels"]:
        if int(row["relevance"]) > 0:
            relevant_by_query[str(row["query_id"])].append(document_index[str(row["document_id"])])

    query_rows = list(dataset["queries"])
    query_ids = [str(row["query_id"]) for row in query_rows]
    if set(query_ids) != set(relevant_by_query):
        raise ValueError("Every query must have at least one positive Qrel")

    rankings_by_route: dict[str, dict[str, list[float | int]]] = {
        route.id: {} for route in routes
    }
    pools = [None] if task.pool_field is None else sorted({str(row[task.pool_field]) for row in query_rows})
    for pool in pools:
        pool_documents = [
            row
            for row in dataset["documents"]
            if pool is None or str(row[task.pool_field]) == pool
        ]
        pool_queries = [
            row
            for row in query_rows
            if pool is None or str(row[task.pool_field]) == pool
        ]
        resources = [
            RetrievalResource(id=str(row["document_id"]), value=None)
            for row in pool_documents
        ]
        corpus_key = SqliteCandidateCache.corpus_key(resources)
        runtime_queries = [task.query_value(row) for row in pool_queries]
        for route in routes:
            cache = SqliteCandidateCache(cache_path, namespace=route.namespace)
            cached = cache.read(
                corpus_key=corpus_key,
                component_id=route.component_id,
                queries=runtime_queries,
                depth=depth,
            )
            if len(cached) != len(pool_queries):
                missing = len(pool_queries) - len(cached)
                raise ValueError(f"Route {route.id!r} is missing {missing} cached query rankings in pool {pool!r}")
            for index, row in enumerate(pool_queries):
                packed: list[float | int] = []
                for hit in cached[index]:
                    packed.extend((document_index[hit.id], hit.score))
                rankings_by_route[route.id][str(row["query_id"])] = packed

    dimension_payload = []
    for dimension in dimensions:
        values = sorted({str(row[dimension.id]) for row in query_rows})
        dimension_payload.append(
            {
                "id": dimension.id,
                "label": dimension.label,
                "values": [
                    {
                        "id": value,
                        "label": (dimension.value_labels or {}).get(value, _title(value)),
                        "count": sum(str(row[dimension.id]) == value for row in query_rows),
                    }
                    for value in values
                ],
            }
        )

    return {
        "task_id": task.id,
        "dataset_id": dataset_id,
        "dataset_version": task.dataset_versions[dataset_id],
        "max_depth": depth,
        "documents": document_ids,
        "dimensions": dimension_payload,
        "queries": [
            {
                "id": query_id,
                "labels": [str(row[dimension.id]) for dimension in dimensions],
                "relevant": sorted(relevant_by_query[query_id]),
            }
            for query_id, row in zip(query_ids, query_rows, strict=True)
        ],
        "routes": [
            {
                "id": route.id,
                "label": route.label,
                "kind": route.kind,
                "score_direction": route.score_direction,
                "source": {
                    "namespace": route.namespace,
                    "component_id": route.component_id,
                },
                "rankings": [rankings_by_route[route.id][query_id] for query_id in query_ids],
            }
            for route in routes
        ],
    }


def write_playground_bundle(benchmarks: Sequence[dict[str, Any]], output: Path) -> None:
    """Write a small manifest and one lazy-loadable gzip file per route."""

    output.mkdir(parents=True, exist_ok=True)
    manifest_benchmarks = []
    expected_files = {output / "manifest.json.gz"}
    for benchmark in benchmarks:
        task_id = str(benchmark["task_id"])
        manifest_routes = []
        for route in benchmark["routes"]:
            route_path = Path(task_id) / f"{route['id']}.json.gz"
            target = output / route_path
            _write_gzip_json(
                {
                    "schema_version": 1,
                    "task_id": task_id,
                    "dataset_id": benchmark["dataset_id"],
                    "dataset_version": benchmark["dataset_version"],
                    "route_id": route["id"],
                    "rankings": route["rankings"],
                },
                target,
            )
            expected_files.add(target)
            manifest_route = {key: value for key, value in route.items() if key != "rankings"}
            manifest_route["path"] = route_path.as_posix()
            manifest_routes.append(manifest_route)
        manifest_benchmark = {key: value for key, value in benchmark.items() if key != "routes"}
        manifest_benchmark["routes"] = manifest_routes
        manifest_benchmarks.append(manifest_benchmark)
    _write_gzip_json(
        {"schema_version": 1, "benchmarks": manifest_benchmarks},
        output / "manifest.json.gz",
    )
    for stale_file in output.rglob("*.json.gz"):
        if stale_file not in expected_files:
            stale_file.unlink()


def _write_gzip_json(payload: Mapping[str, Any], output: Path) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        stream.write(encoded)


def _title(value: str) -> str:
    abbreviations = {"ai": "AI", "qa": "Q&A"}
    return " ".join(
        abbreviations.get(word, word.capitalize())
        for word in value.replace("/", " / ").replace("_", " ").split()
    )
