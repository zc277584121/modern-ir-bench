"""Load and evaluate immutable rankings without pretending they are Solutions."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modern_ir_bench.core.provenance import RunProvenance
from modern_ir_bench.core.result import RunReport
from modern_ir_bench.tasks import RankedRetrieval


@dataclass(frozen=True)
class RankingArtifact:
    manifest: Mapping[str, Any]
    rankings: Mapping[str, Mapping[str, tuple[str, ...]]]

    @classmethod
    def load(cls, directory: Path) -> RankingArtifact:
        directory = directory.resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported ranking artifact schema")
        if not str(manifest.get("artifact_type", "")).endswith("top-k-rankings"):
            raise ValueError("Manifest does not describe a Top-K ranking artifact")
        if len(manifest["solutions"]) != len(set(manifest["solutions"])):
            raise ValueError("Ranking artifact manifest contains duplicate Solution ids")
        ranking_path = directory / "rankings.jsonl.gz"
        checksum = hashlib.sha256(ranking_path.read_bytes()).hexdigest()
        if checksum != manifest["rankings_sha256"]:
            raise ValueError("Ranking artifact checksum does not match its manifest")
        by_solution: dict[str, dict[str, tuple[str, ...]]] = {}
        seen: set[tuple[str, str]] = set()
        with gzip.open(ranking_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                solution_id = str(row["solution_id"])
                query_id = str(row["query_id"])
                pair = solution_id, query_id
                if pair in seen:
                    raise ValueError(f"Duplicate published ranking: {pair}")
                seen.add(pair)
                ranked_ids = tuple(str(value) for value in row["ranked_doc_ids"])
                if len(ranked_ids) != len(set(ranked_ids)):
                    raise ValueError(f"Published ranking contains duplicate documents: {pair}")
                if len(ranked_ids) != int(manifest["top_k"]):
                    raise ValueError(f"Published ranking has the wrong depth: {pair}")
                by_solution.setdefault(solution_id, {})[query_id] = ranked_ids
        expected_solutions = set(manifest["solutions"])
        if set(by_solution) != expected_solutions:
            raise ValueError("Ranking artifact Solution set does not match its manifest")
        if len(seen) != int(manifest["ranking_rows"]):
            raise ValueError("Ranking artifact row count does not match its manifest")
        expected_queries = int(manifest["queries"])
        if any(len(rankings) != expected_queries for rankings in by_solution.values()):
            raise ValueError("Ranking artifact query count does not match its manifest")
        return cls(manifest=manifest, rankings=by_solution)

    def evaluate(
        self,
        *,
        task: RankedRetrieval,
        dataset_id: str,
        dataset: Any,
        dataset_version: str,
        provenance: RunProvenance,
        solution_metadata: Mapping[str, Mapping[str, Any]],
    ) -> RunReport:
        if set(solution_metadata) != set(self.rankings):
            raise ValueError("Solution metadata does not match the ranking artifact")
        report = RunReport()
        task.validate_dataset(dataset)
        for solution_id, rankings in self.rankings.items():
            metadata = solution_metadata[solution_id]
            observations = task.observations_from_rankings(
                dataset,
                rankings,
                validate=False,
            )
            report.add_evaluation(
                task_id=task.id,
                task_title=task.title,
                task_description=task.description,
                task_version=task.version,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                solution_id=solution_id,
                solution_title=str(metadata["title"]),
                solution_description=str(metadata["description"]),
                primary_metric_id=task.metrics.primary.id,
                metrics=task.metrics.evaluate(observations),
                observations=observations,
                provenance=provenance,
                status="historical-artifact",
            )
            report.solutions[solution_id].update(metadata)
        return report
