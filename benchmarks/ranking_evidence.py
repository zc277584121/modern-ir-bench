"""Verified per-Solution runs and immutable ranking releases."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

from modern_ir_bench.core.result import RunReport
from modern_ir_bench.results import RankingArtifact
from modern_ir_bench.tasks import RankedRetrieval

RUN_ARTIFACT_TYPE = "live-ranking-solution-run"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _ranking_depth(observations: Dataset) -> int:
    depths = set(map(len, observations["ranked_ids"]))
    if len(depths) != 1:
        raise ValueError("Observation rankings do not have one consistent depth")
    return next(iter(depths))


def _rankings(observations: Dataset) -> dict[str, tuple[str, ...]]:
    rankings = {
        str(row["query_id"]): tuple(str(document_id) for document_id in row["ranked_ids"])
        for row in observations
    }
    if len(rankings) != len(observations):
        raise ValueError("Observation artifact contains duplicate query ids")
    return rankings


def write_run_artifacts(
    report: RunReport,
    *,
    output_root: Path,
    release: str,
    dataset_repository: str,
    dataset_revision: str,
) -> list[Path]:
    """Write one independently verifiable directory per evaluated Solution."""

    directories = []
    for solution_id, solution in report.solutions.items():
        records = [row for row in report.records if row["solution_id"] == solution_id]
        if not records:
            raise ValueError(f"No metric records for {solution_id}")
        source_commits = {str(row["source_commit"]) for row in records}
        if len(source_commits) != 1:
            raise ValueError(f"Run contains mixed source commits for {solution_id}")
        source_commit = next(iter(source_commits))
        observation_keys = {str(row["observation_key"]) for row in records}
        if len(observation_keys) != 1:
            raise ValueError(
                "A run artifact represents one Solution on one Dataset; "
                f"found {len(observation_keys)} evaluations for {solution_id}"
            )
        observations = report.observations[next(iter(observation_keys))]
        directory = output_root / solution_id / source_commit
        directory.mkdir(parents=True, exist_ok=True)
        results_path = directory / "results.json"
        observations_path = directory / "observations.parquet"
        results_path.write_text(
            json.dumps(
                {"release": release, "results": records, "solution": solution},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        observations.to_parquet(observations_path)
        manifest = {
            "schema_version": 1,
            "artifact_type": RUN_ARTIFACT_TYPE,
            "release": release,
            "dataset": {"repository": dataset_repository, "revision": dataset_revision},
            "solution": solution_id,
            "source_commit": source_commit,
            "queries": len(observations),
            "top_k": _ranking_depth(observations),
            "results_sha256": _sha256(results_path),
            "observations_sha256": _sha256(observations_path),
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        directories.append(directory)
    return directories


def load_run_artifact(directory: Path) -> tuple[dict[str, Any], dict[str, Any], Dataset]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != RUN_ARTIFACT_TYPE:
        raise ValueError(f"Unsupported Solution run artifact: {directory}")
    results_path = directory / "results.json"
    observations_path = directory / "observations.parquet"
    if _sha256(results_path) != manifest["results_sha256"]:
        raise ValueError(f"Run results checksum mismatch: {directory}")
    if _sha256(observations_path) != manifest["observations_sha256"]:
        raise ValueError(f"Run observations checksum mismatch: {directory}")
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    if saved.get("release") != manifest["release"]:
        raise ValueError(f"Run release mismatch: {directory}")
    if saved.get("solution", {}).get("id") != manifest["solution"]:
        raise ValueError(f"Run Solution mismatch: {directory}")
    observations = Dataset.from_parquet(str(observations_path))
    if len(observations) != int(manifest["queries"]):
        raise ValueError(f"Run query count mismatch: {directory}")
    if _ranking_depth(observations) != int(manifest["top_k"]):
        raise ValueError(f"Run ranking depth mismatch: {directory}")
    return manifest, saved, observations


def _existing_release(directory: Path) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return {}, {}
    artifact = RankingArtifact.load(directory)
    source_runs = dict(artifact.manifest.get("source_runs", {}))
    for solution_id in artifact.rankings:
        if solution_id in source_runs:
            continue
        source_runs[solution_id] = {
            key: value
            for key, value in {
                "origin": artifact.manifest["artifact_type"],
                "published_at": artifact.manifest.get("published_at"),
                "source_commit": artifact.manifest.get("source_commit"),
            }.items()
            if value is not None
        }
    return {solution: dict(rows) for solution, rows in artifact.rankings.items()}, source_runs


def merge_release(
    *,
    task: RankedRetrieval,
    dataset_id: str,
    dataset_repository: str,
    dataset_revision: str,
    run_directories: Sequence[Path],
    output: Path,
    remove_solutions: Sequence[str] = (),
) -> None:
    """Merge verified Solution runs into one immutable Top-K ranking release."""

    if not run_directories and not remove_solutions:
        raise ValueError("At least one verified Solution run or Solution removal is required")
    dataset = task.datasets[dataset_id]
    task.validate_dataset(dataset)
    existing_manifest_path = output / "manifest.json"
    if existing_manifest_path.is_file():
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("release") != dataset_id:
            raise ValueError("Existing output belongs to a different release")
        if existing_manifest.get("dataset") != {
            "repository": dataset_repository,
            "revision": dataset_revision,
        }:
            raise ValueError("Existing output belongs to a different Dataset")
        if int(existing_manifest.get("top_k", -1)) != task.top_k:
            raise ValueError("Existing output uses a different ranking depth")
    all_rankings, source_runs = _existing_release(output)
    top_k = task.top_k

    unknown_removals = sorted(set(remove_solutions) - set(all_rankings))
    if unknown_removals:
        raise ValueError(f"Cannot remove unknown Solutions: {', '.join(unknown_removals)}")
    for solution_id in set(remove_solutions):
        all_rankings.pop(solution_id)
        source_runs.pop(solution_id, None)

    for directory in run_directories:
        manifest, saved, observations = load_run_artifact(directory)
        if manifest["release"] != dataset_id:
            raise ValueError(f"Run uses a different release: {directory}")
        if manifest["dataset"] != {"repository": dataset_repository, "revision": dataset_revision}:
            raise ValueError(f"Run uses a different Dataset: {directory}")
        if int(manifest["top_k"]) != top_k:
            raise ValueError(f"Run uses a different ranking depth: {directory}")
        solution_id = str(manifest["solution"])
        rankings = _rankings(observations)
        canonical = task.observations_from_rankings(dataset, rankings)
        computed = {metric.metric_id: metric.value for metric in task.metrics.evaluate(canonical)}
        recorded = {str(row["metric_id"]): float(row["value"]) for row in saved["results"]}
        if {key: round(value, 8) for key, value in computed.items()} != recorded:
            raise ValueError(f"Run aggregate metrics do not match its rankings: {directory}")
        all_rankings[solution_id] = rankings
        first = saved["results"][0]
        source_runs[solution_id] = {
            "source_commit": str(manifest["source_commit"]),
            "source_module": str(first["source_module"]),
            "source_path": str(first["source_path"]),
            "source_line": int(first["source_line"]),
            "created_at": str(first["created_at"]),
            "origin": "verified-live-run",
        }

    if not all_rankings:
        raise ValueError("A ranking release requires at least one Solution run")
    query_ids = set(dataset["queries"]["query_id"])
    for solution_id, rankings in all_rankings.items():
        if set(rankings) != query_ids:
            raise ValueError(f"Rankings do not cover every query for {solution_id}")

    output.mkdir(parents=True, exist_ok=True)
    ranking_path = output / "rankings.jsonl.gz"
    temporary = ranking_path.with_suffix(".tmp.gz")
    with (
        temporary.open("wb") as raw_stream,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as stream,
    ):
        for solution_id in sorted(all_rankings):
            for query_id in sorted(all_rankings[solution_id]):
                stream.write(
                    json.dumps(
                        {
                            "solution_id": solution_id,
                            "query_id": query_id,
                            "ranked_doc_ids": list(all_rankings[solution_id][query_id]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    temporary.replace(ranking_path)
    manifest = {
        "schema_version": 1,
        "artifact_type": "live-top-k-rankings",
        "release": dataset_id,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"repository": dataset_repository, "revision": dataset_revision},
        "solutions": sorted(all_rankings),
        "source_runs": {solution_id: source_runs[solution_id] for solution_id in sorted(all_rankings)},
        "queries": len(query_ids),
        "ranking_rows": len(query_ids) * len(all_rankings),
        "top_k": top_k,
        "rankings_sha256": _sha256(ranking_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_release_readme(output / "README.md", task.title, sorted(all_rankings))


def _write_release_readme(path: Path, task_title: str, solution_ids: list[str]) -> None:
    bullets = "\n".join(f"- `{solution_id}`" for solution_id in solution_ids)
    path.write_text(
        f"# {task_title} result evidence\n\n"
        "This directory stores immutable Top-K document rankings for every published Solution. "
        "Aggregate metrics and leaderboard slices are recomputed from these rankings and the pinned Dataset.\n\n"
        "## Solutions\n\n"
        f"{bullets}\n\n"
        "`manifest.json` pins the Dataset, source runs, ranking depth, row count, and checksum.\n",
        encoding="utf-8",
    )
