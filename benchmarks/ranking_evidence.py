"""Verified per-Solution runs and immutable ranking releases."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

from benchmarks.languages import BENCHMARK_LANGUAGES, validate_solution_languages
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
        str(row["query_id"]): tuple(str(document_id) for document_id in row["ranked_ids"]) for row in observations
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


def validate_artifact_scope(
    manifest: Mapping[str, Any],
    *,
    task: RankedRetrieval,
    dataset_id: str,
    dataset_repository: str,
    dataset_revision: str,
) -> None:
    """Check evidence identity before using its rankings or aggregate scores."""
    if manifest.get("release") != dataset_id:
        raise ValueError("Evidence uses a different release")
    if manifest.get("dataset") != {"repository": dataset_repository, "revision": dataset_revision}:
        raise ValueError("Evidence uses a different Dataset")
    if int(manifest.get("top_k", -1)) != task.top_k:
        raise ValueError("Evidence uses a different ranking depth")
    dataset = task.datasets[dataset_id]
    if int(manifest.get("queries", -1)) != len(dataset["queries"]):
        raise ValueError("Evidence does not cover the complete Dataset scope")
    if "languages" in manifest and set(manifest["languages"]) != set(dataset["queries"]["language"]):
        raise ValueError("Evidence language scope does not match its Dataset")


def validate_language_slice(full: Mapping[str, Dataset], sliced: Mapping[str, Dataset], language: str) -> None:
    """Require the same query, candidate, and relevance rows as the bilingual scope."""
    query_ids = {row["query_id"] for row in full["queries"] if row["language"] == language}
    for split, keys in (
        ("queries", ["query_id"]),
        ("documents", ["document_id"]),
        ("qrels", ["query_id", "document_id"]),
    ):
        expected = full[split].filter(
            (lambda row: row["query_id"] in query_ids)
            if split == "qrels"
            else (lambda row: row["language"] == language)
        )
        actual = sliced[split]
        if set(expected.column_names) != set(actual.column_names):
            raise ValueError(f"Language slice {split} differ from the bilingual Dataset")
        # Dataset indices must be applied before comparing selected rows.
        expected_table = expected.sort(keys).with_format("arrow")[:]
        actual_table = actual.select_columns(expected.column_names).sort(keys).with_format("arrow")[:]
        if not expected_table.equals(actual_table):
            raise ValueError(f"Language slice {split} differ from the bilingual Dataset")


def _source_artifact(directory: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest_sha256": _sha256(directory / "manifest.json"),
        **{
            key: manifest[key]
            for key in (
                "artifact_type",
                "release",
                "dataset",
                "queries",
                "top_k",
                "rankings_sha256",
                "results_sha256",
                "observations_sha256",
            )
            if key in manifest
        },
    }


def _verified_run(
    directory: Path,
    *,
    task: RankedRetrieval,
    dataset_id: str,
    dataset_repository: str,
    dataset_revision: str,
) -> tuple[str, dict[str, tuple[str, ...]], dict[str, Any]]:
    manifest, saved, observations = load_run_artifact(directory)
    validate_artifact_scope(
        manifest,
        task=task,
        dataset_id=dataset_id,
        dataset_repository=dataset_repository,
        dataset_revision=dataset_revision,
    )
    solution_id = str(manifest["solution"])
    rankings = _rankings(observations)
    canonical = task.observations_from_rankings(task.datasets[dataset_id], rankings)
    computed = {metric.metric_id: round(metric.value, 8) for metric in task.metrics.evaluate(canonical)}
    records = saved["results"]
    recorded = {str(row["metric_id"]): float(row["value"]) for row in records}
    if len(recorded) != len(records) or computed != recorded:
        raise ValueError(f"Run aggregate metrics do not match its rankings: {directory}")
    for row in records:
        expected = {
            "solution_id": solution_id,
            "task_id": task.id,
            "task_version": task.version,
            "dataset_id": dataset_id,
            "dataset_version": task.dataset_versions[dataset_id],
            "source_commit": manifest["source_commit"],
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Run metric identity does not match its manifest: {directory}")
    provenance_keys = ("source_commit", "source_module", "source_path", "source_line", "created_at")
    first = records[0]
    if any(any(row[key] != first[key] for key in provenance_keys) for row in records):
        raise ValueError(f"Run contains inconsistent provenance: {directory}")
    return (
        solution_id,
        rankings,
        {
            **{key: first[key] for key in provenance_keys},
            "origin": "verified-live-run",
            "source_artifact": _source_artifact(directory, manifest),
        },
    )


def _derive_language_rankings(
    directory: Path,
    *,
    source_task: RankedRetrieval,
    source_dataset_id: str,
    dataset_repository: str,
    dataset_revision: str,
    language: str,
    solution_ids: Sequence[str] | None,
) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    validate_artifact_scope(
        manifest,
        task=source_task,
        dataset_id=source_dataset_id,
        dataset_repository=dataset_repository,
        dataset_revision=dataset_revision,
    )
    if manifest["artifact_type"] == RUN_ARTIFACT_TYPE:
        solution_id, rankings, source = _verified_run(
            directory,
            task=source_task,
            dataset_id=source_dataset_id,
            dataset_repository=dataset_repository,
            dataset_revision=dataset_revision,
        )
        all_rankings, sources = {solution_id: rankings}, {solution_id: source}
    else:
        all_rankings, sources = _existing_release(directory)
    dataset = source_task.datasets[source_dataset_id]
    selected_queries = {row["query_id"] for row in dataset["queries"] if row["language"] == language}
    selected = set(all_rankings) if solution_ids is None else set(all_rankings).intersection(solution_ids)
    derived, provenance = {}, {}
    for solution_id in sorted(selected):
        rankings = all_rankings[solution_id]
        source_task.observations_from_rankings(dataset, rankings, validate=False)
        if any(len(ranked) != source_task.top_k for ranked in rankings.values()):
            raise ValueError(f"Source ranking depth mismatch: {solution_id}")
        derived[solution_id] = {query_id: rankings[query_id] for query_id in sorted(selected_queries)}
        provenance[solution_id] = {
            **sources[solution_id],
            "derivation": {
                "operation": "language-slice",
                "language": language,
                "source_artifact": _source_artifact(directory, manifest),
            },
        }
    return derived, provenance


def merge_release(
    *,
    task: RankedRetrieval,
    dataset_id: str,
    dataset_repository: str,
    dataset_revision: str,
    run_directories: Sequence[Path],
    output: Path,
    solution_languages: Mapping[str, Sequence[str]],
    remove_solutions: Sequence[str] = (),
    derive_from: Sequence[Path] = (),
    derive_solutions: Sequence[str] | None = None,
    source_task: RankedRetrieval | None = None,
    source_dataset_id: str | None = None,
) -> None:
    """Merge verified Solution runs into one immutable Top-K ranking release."""

    if not run_directories and not remove_solutions and not derive_from:
        raise ValueError("At least one verified Solution run or Solution removal is required")
    dataset = task.datasets[dataset_id]
    task.validate_dataset(dataset)
    languages = sorted(set(dataset["queries"]["language"]))
    if derive_from:
        if len(languages) != 1 or source_task is None or source_dataset_id is None:
            raise ValueError("Language derivation requires a single-language target and a complete source Task")
        if dataset_id != f"{source_dataset_id}-language-{languages[0]}":
            raise ValueError("Derived release id does not match its parent and language")
        if source_task.pool_field != "language" or task.pool_field != "language":
            raise ValueError("Language derivation requires independent language retrieval pools")
        source_dataset = source_task.datasets[source_dataset_id]
        source_task.validate_dataset(source_dataset)
        if set(source_dataset["queries"]["language"]) != set(BENCHMARK_LANGUAGES):
            raise ValueError("Language derivation requires a complete bilingual source")
        validate_language_slice(source_dataset, dataset, languages[0])
        if (
            source_task.id != task.id
            or source_task.version != task.version
            or source_task.top_k != task.top_k
            or source_task.dataset_versions[source_dataset_id] != task.dataset_versions[dataset_id]
        ):
            raise ValueError("Source and target Task definitions differ")
    elif derive_solutions is not None:
        raise ValueError("Solution selection requires --derive-from")
    existing_manifest_path = output / "manifest.json"
    if existing_manifest_path.is_file():
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        validate_artifact_scope(
            existing_manifest,
            task=task,
            dataset_id=dataset_id,
            dataset_repository=dataset_repository,
            dataset_revision=dataset_revision,
        )
    all_rankings, source_runs = _existing_release(output)
    top_k = task.top_k

    unknown_removals = sorted(set(remove_solutions) - set(all_rankings))
    if unknown_removals:
        raise ValueError(f"Cannot remove unknown Solutions: {', '.join(unknown_removals)}")
    for solution_id in set(remove_solutions):
        all_rankings.pop(solution_id)
        source_runs.pop(solution_id, None)

    incoming_ids = set()
    for directory in (*run_directories, *derive_from):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        ids = {manifest["solution"]} if "solution" in manifest else set(manifest["solutions"])
        if directory in derive_from and derive_solutions is not None:
            ids.intersection_update(derive_solutions)
        if incoming_ids.intersection(ids):
            raise ValueError(f"Multiple incoming artifacts for the same Solution: {sorted(incoming_ids & ids)}")
        incoming_ids.update(ids)
    validate_solution_languages(sorted(set(all_rankings) | incoming_ids), solution_languages, languages)

    for directory in run_directories:
        solution_id, rankings, source = _verified_run(
            directory,
            task=task,
            dataset_id=dataset_id,
            dataset_repository=dataset_repository,
            dataset_revision=dataset_revision,
        )
        all_rankings[solution_id] = rankings
        source_runs[solution_id] = source

    derived_ids = set()
    for directory in derive_from:
        derived, sources = _derive_language_rankings(
            directory,
            source_task=source_task,
            source_dataset_id=source_dataset_id,
            dataset_repository=dataset_repository,
            dataset_revision=dataset_revision,
            language=languages[0],
            solution_ids=derive_solutions,
        )
        derived_ids.update(derived)
        all_rankings.update(derived)
        source_runs.update(sources)
    if derive_solutions is not None and set(derive_solutions) != derived_ids:
        raise ValueError(
            f"Requested derivation Solutions are missing from the sources: {sorted(set(derive_solutions) - derived_ids)}"
        )

    if not all_rankings:
        raise ValueError("A ranking release requires at least one Solution run")
    query_ids = set(dataset["queries"]["query_id"])
    for solution_id, rankings in all_rankings.items():
        if set(rankings) != query_ids:
            raise ValueError(f"Rankings do not cover every query for {solution_id}")
        if any(len(ranked) != top_k for ranked in rankings.values()):
            raise ValueError(f"Rankings have the wrong depth for {solution_id}")
        task.observations_from_rankings(dataset, rankings, validate=False)

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
        "languages": languages,
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
