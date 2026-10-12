"""Build the immutable multi-Solution PDF ranking release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

from benchmarks.pdf_retrieval.artifacts import load_run_artifact
from benchmarks.pdf_retrieval.definition import DATASET_ID, DATASET_REVISION, RELEASE_ID, build_task
from modern_ir_bench.results import RankingArtifact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/modern-ir-pdf-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rankings(observations: Dataset) -> dict[str, tuple[str, ...]]:
    rankings = {
        str(row["query_id"]): tuple(str(document_id) for document_id in row["ranked_ids"])
        for row in observations
    }
    if len(rankings) != len(observations):
        raise ValueError("Observation artifact contains duplicate query ids")
    return rankings


def _existing_release(directory: Path) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return {}, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("artifact_type", "")).endswith("top-k-rankings"):
        artifact = RankingArtifact.load(directory)
        return {solution: dict(rows) for solution, rows in artifact.rankings.items()}, dict(
            manifest.get("source_runs", {})
        )

    if manifest.get("artifact_type") != "live-pdf-retrieval-observations":
        raise ValueError("Existing PDF result directory has an unsupported artifact type")
    observations_path = directory / "observations.parquet"
    results_path = directory / "results.json"
    if _sha256(observations_path) != manifest["observations_sha256"]:
        raise ValueError("Legacy PDF observations checksum mismatch")
    if _sha256(results_path) != manifest["results_sha256"]:
        raise ValueError("Legacy PDF results checksum mismatch")
    solution_id = str(manifest["solution"])
    observations = Dataset.from_parquet(str(observations_path))
    return {solution_id: _rankings(observations)}, {
        solution_id: {
            "source_commit": str(manifest["source_commit"]),
            "origin": "legacy-live-observations",
        }
    }


def build_release(*, run_directories: list[Path], output: Path = DEFAULT_OUTPUT) -> None:
    task = build_task()
    dataset = task.datasets[RELEASE_ID]
    task.validate_dataset(dataset)
    all_rankings, source_runs = _existing_release(output)
    top_k = task.top_k

    for directory in run_directories:
        manifest, saved, observations = load_run_artifact(directory)
        if manifest["dataset"] != {"repository": DATASET_ID, "revision": DATASET_REVISION}:
            raise ValueError(f"Run uses a different PDF Dataset: {directory}")
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
        raise ValueError("PDF release requires at least one Solution ranking")
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
        "release": RELEASE_ID,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"repository": DATASET_ID, "revision": DATASET_REVISION},
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
    _write_readme(output / "README.md", sorted(all_rankings))


def _write_readme(path: Path, solution_ids: list[str]) -> None:
    bullets = "\n".join(f"- `{solution_id}`" for solution_id in solution_ids)
    path.write_text(
        "# Modern IR PDF Retrieval v1 result evidence\n\n"
        "This directory stores immutable Top-10 document rankings for every published PDF Solution. "
        "Aggregate metrics and leaderboard slices are always recomputed from these rankings and the pinned Dataset.\n\n"
        "## Solutions\n\n"
        f"{bullets}\n\n"
        "`manifest.json` pins the Dataset, source runs, ranking depth, row count, and checksum.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], type=Path, help="Verified run artifact directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    build_release(run_directories=arguments.run, output=arguments.output)


if __name__ == "__main__":
    main()
