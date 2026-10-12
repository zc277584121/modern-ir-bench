"""Validated per-Solution run artifacts for PDF Retrieval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from modern_ir_bench.core.result import RunReport


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
            raise ValueError(f"Run contains mixed observations for {solution_id}")
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
            "artifact_type": "live-pdf-solution-run",
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
    if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "live-pdf-solution-run":
        raise ValueError(f"Unsupported PDF run artifact: {directory}")
    results_path = directory / "results.json"
    observations_path = directory / "observations.parquet"
    if _sha256(results_path) != manifest["results_sha256"]:
        raise ValueError(f"Run results checksum mismatch: {directory}")
    if _sha256(observations_path) != manifest["observations_sha256"]:
        raise ValueError(f"Run observations checksum mismatch: {directory}")
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    observations = Dataset.from_parquet(str(observations_path))
    if len(observations) != int(manifest["queries"]):
        raise ValueError(f"Run query count mismatch: {directory}")
    if _ranking_depth(observations) != int(manifest["top_k"]):
        raise ValueError(f"Run ranking depth mismatch: {directory}")
    return manifest, saved, observations


def _ranking_depth(observations: Dataset) -> int:
    depths = set(map(len, observations["ranked_ids"]))
    if len(depths) != 1:
        raise ValueError("Observation rankings do not have one consistent depth")
    return next(iter(depths))
