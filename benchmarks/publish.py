"""Build the complete Space payload from immutable result evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

from benchmarks.pdf_retrieval.definition import (
    DATASET_REVISION as PDF_DATASET_REVISION,
)
from benchmarks.pdf_retrieval.definition import (
    RELEASE_ID as PDF_RELEASE_ID,
)
from benchmarks.pdf_retrieval.definition import (
    build_task as build_pdf_task,
)
from benchmarks.pdf_retrieval.solution import SOLUTION_ID as PDF_SOLUTION_ID
from benchmarks.pdf_retrieval.solution import solution_metadata as pdf_solution_metadata
from benchmarks.ranked_retrieval.definition import (
    DATASET_REVISION as TEXT_DATASET_REVISION,
)
from benchmarks.ranked_retrieval.definition import (
    RELEASE_ID as TEXT_RELEASE_ID,
)
from benchmarks.ranked_retrieval.definition import (
    build_task as build_text_task,
)
from benchmarks.ranked_retrieval.solutions import solution_metadata as text_solution_metadata
from benchmarks.reporting import Dimension, build_retrieval_breakdowns
from modern_ir_bench import RunProvenance
from modern_ir_bench.core.result import RunReport
from modern_ir_bench.exporters import build_space_payload
from modern_ir_bench.results import RankingArtifact

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_RESULT_ROOT = PROJECT_ROOT / "results/modern-ir-v2.2"
PDF_RESULT_ROOT = PROJECT_ROOT / "results/modern-ir-pdf-v1"
SPACE_RESULTS = PROJECT_ROOT / "space/data/results.json"

TEXT_QUERY_DIMENSIONS = (
    Dimension("language", "Language", (("zh", "Chinese"), ("en", "English"))),
    Dimension(
        "query_intent",
        "Intent",
        (
            ("understand", "Understand"),
            ("act", "Act"),
            ("lookup", "Lookup"),
            ("decide", "Decide"),
            ("verify", "Verify"),
            ("synthesize", "Synthesize"),
        ),
    ),
    Dimension(
        "expression",
        "Expression",
        (
            ("natural_question", "Natural question"),
            ("search_phrase", "Search phrase"),
            ("contextual_request", "Contextual request"),
        ),
    ),
    Dimension(
        "constraint_level",
        "Constraint level",
        (("single", "Single"), ("compound", "Compound"), ("light", "Light")),
    ),
)
TEXT_DOCUMENT_DIMENSIONS = tuple(
    Dimension(field, label)
    for field, label in (
        ("language", "Language"),
        ("domain", "Domain"),
        ("form", "Form"),
        ("length", "Length"),
    )
)
PDF_QUERY_DIMENSIONS = (Dimension("language", "Language", (("zh", "Chinese"), ("en", "English"))),)
PDF_DOCUMENT_DIMENSIONS = (
    Dimension("transform_tier", "Capture difficulty", (("easy", "Easy"), ("medium", "Medium"), ("hard", "Hard"))),
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository_url(remote: str) -> str:
    if remote.startswith("git@github.com:"):
        remote = f"https://github.com/{remote.removeprefix('git@github.com:')}"
    return remote.removesuffix(".git")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _labels(rows: Dataset, *, id_field: str, fields: tuple[str, ...]) -> dict[str, dict[str, str]]:
    return {str(row[id_field]): {field: str(row[field]) for field in fields} for row in rows}


def _relevance(qrels: Dataset) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        if int(row["relevance"]) > 0:
            relevant[str(row["query_id"])].add(str(row["document_id"]))
    return dict(relevant)


def _rankings(observations: Dataset) -> dict[str, tuple[str, ...]]:
    rankings = {
        str(row["query_id"]): tuple(str(document_id) for document_id in row["ranked_ids"]) for row in observations
    }
    if len(rankings) != len(observations):
        raise ValueError("Observation artifact contains duplicate query ids")
    return rankings


def _text_release(repository_url: str, commit: str) -> tuple[RunReport, list[dict[str, Any]]]:
    task = build_text_task()
    dataset = task.datasets[TEXT_RELEASE_ID]
    artifact = RankingArtifact.load(TEXT_RESULT_ROOT)
    provenance = RunProvenance(
        source_commit=commit,
        source_module="results.modern_ir_v2_2",
        source_path="results/modern-ir-v2.2/README.md",
        source_line=1,
        repository_url=repository_url,
        created_at=str(artifact.manifest.get("published_at", datetime.now(timezone.utc).isoformat())),
    )
    metadata = text_solution_metadata(repository_url, commit)
    result_url = f"{repository_url}/tree/{commit}/results/modern-ir-v2.2"
    for values in metadata.values():
        values["result_url"] = result_url
    report = artifact.evaluate(
        task=task,
        dataset_id=TEXT_RELEASE_ID,
        dataset=dataset,
        dataset_version=TEXT_DATASET_REVISION[:12],
        provenance=provenance,
        solution_metadata=metadata,
    )
    breakdowns = build_retrieval_breakdowns(
        task_id=task.id,
        dataset_id=TEXT_RELEASE_ID,
        dataset_version=TEXT_DATASET_REVISION[:12],
        rankings=artifact.rankings,
        relevant_by_query=_relevance(dataset["qrels"]),
        query_labels=_labels(
            dataset["queries"],
            id_field="query_id",
            fields=tuple(dimension.id for dimension in TEXT_QUERY_DIMENSIONS),
        ),
        document_labels=_labels(
            dataset["documents"],
            id_field="document_id",
            fields=tuple(dimension.id for dimension in TEXT_DOCUMENT_DIMENSIONS),
        ),
        query_dimensions=TEXT_QUERY_DIMENSIONS,
        document_dimensions=TEXT_DOCUMENT_DIMENSIONS,
    )
    return report, breakdowns


def _validate_pdf_artifact() -> tuple[dict[str, Any], Dataset]:
    manifest = json.loads((PDF_RESULT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    result_path = PDF_RESULT_ROOT / "results.json"
    observation_path = PDF_RESULT_ROOT / "observations.parquet"
    if _sha256(result_path) != manifest["results_sha256"]:
        raise ValueError("PDF results checksum does not match its manifest")
    if _sha256(observation_path) != manifest["observations_sha256"]:
        raise ValueError("PDF observations checksum does not match its manifest")
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    observations = Dataset.from_parquet(str(observation_path))
    if len(observations) != int(manifest["queries"]):
        raise ValueError("PDF observation count does not match its manifest")
    if set(map(len, observations["ranked_ids"])) != {int(manifest["top_k"])}:
        raise ValueError("PDF rankings do not match the declared depth")
    return saved, observations


def _pdf_release(repository_url: str, commit: str) -> tuple[RunReport, list[dict[str, Any]]]:
    saved, observations = _validate_pdf_artifact()
    task = build_pdf_task()
    dataset = task.datasets[PDF_RELEASE_ID]
    task.validate_dataset(dataset)
    metadata = pdf_solution_metadata(repository_url, commit)
    metadata["result_url"] = f"{repository_url}/tree/{commit}/results/modern-ir-pdf-v1"
    first = saved["results"][0]
    provenance = RunProvenance(
        source_commit=str(first["source_commit"]),
        source_module=str(first["source_module"]),
        source_path=str(first["source_path"]),
        source_line=int(first["source_line"]),
        repository_url=repository_url,
        created_at=str(first["created_at"]),
    )
    report = RunReport()
    report.add_evaluation(
        task_id=task.id,
        task_title=task.title,
        task_description=task.description,
        task_version=task.version,
        dataset_id=PDF_RELEASE_ID,
        dataset_version=PDF_DATASET_REVISION[:12],
        solution_id=PDF_SOLUTION_ID,
        solution_title=str(metadata["title"]),
        solution_description=str(metadata["description"]),
        primary_metric_id=task.metrics.primary.id,
        metrics=task.metrics.evaluate(observations),
        observations=observations,
        provenance=provenance,
        status=str(first["status"]),
    )
    report.solutions[PDF_SOLUTION_ID].update(metadata)
    saved_values = {str(row["metric_id"]): float(row["value"]) for row in saved["results"]}
    computed_values = {str(row["metric_id"]): float(row["value"]) for row in report.records}
    if saved_values != computed_values:
        raise ValueError("PDF aggregate metrics do not match the preserved observations")

    solution_rankings = _rankings(observations)
    breakdowns = build_retrieval_breakdowns(
        task_id=task.id,
        dataset_id=PDF_RELEASE_ID,
        dataset_version=PDF_DATASET_REVISION[:12],
        rankings={PDF_SOLUTION_ID: solution_rankings},
        relevant_by_query=_relevance(dataset["qrels"]),
        query_labels=_labels(
            dataset["queries"],
            id_field="query_id",
            fields=tuple(dimension.id for dimension in PDF_QUERY_DIMENSIONS),
        ),
        document_labels=_labels(
            dataset["documents"],
            id_field="document_id",
            fields=tuple(dimension.id for dimension in PDF_DOCUMENT_DIMENSIONS),
        ),
        query_dimensions=PDF_QUERY_DIMENSIONS,
        document_dimensions=PDF_DOCUMENT_DIMENSIONS,
    )
    return report, breakdowns


def build_payload() -> dict[str, object]:
    """Build one deterministic payload; no task mutates an existing Space file."""

    if _git("status", "--porcelain"):
        raise RuntimeError("Commit source changes before generating immutable GitHub links")
    commit = _git(
        "log",
        "-1",
        "--format=%H",
        "--",
        "benchmarks",
        "src",
        "docs/solutions",
        "results",
    )
    repository_url = _repository_url(_git("remote", "get-url", "origin"))
    text_report, text_breakdowns = _text_release(repository_url, commit)
    pdf_report, pdf_breakdowns = _pdf_release(repository_url, commit)
    text_report.extend(pdf_report)
    return build_space_payload(
        text_report,
        release="modern-ir-bench-public-20261011",
        counts={"documents": 5000, "queries": 1000, "qrels": 6575},
        languages=("Chinese", "English"),
        breakdowns=(*text_breakdowns, *pdf_breakdowns),
        notice=(
            "Ranked Retrieval v2.2 is recomputed from reviewed historical Top-10 rankings. "
            "PDF Retrieval v1 is recomputed from preserved live observations."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when the committed Space payload is stale")
    arguments = parser.parse_args()
    rendered = json.dumps(build_payload(), ensure_ascii=False, indent=2) + "\n"
    if arguments.check:
        if not SPACE_RESULTS.is_file() or SPACE_RESULTS.read_text(encoding="utf-8") != rendered:
            raise SystemExit("space/data/results.json is stale; run `uv run python -m benchmarks.publish`")
        return
    SPACE_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    SPACE_RESULTS.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
