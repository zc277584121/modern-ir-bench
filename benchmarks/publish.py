"""Build the complete Space payload from immutable result evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset

from benchmarks.languages import BENCHMARK_LANGUAGES, validate_solution_languages
from benchmarks.pdf_retrieval.definition import DATASET_ID as PDF_DATASET_ID
from benchmarks.pdf_retrieval.definition import (
    DATASET_REVISION as PDF_DATASET_REVISION,
)
from benchmarks.pdf_retrieval.definition import (
    build_task as build_pdf_task,
)
from benchmarks.pdf_retrieval.definition import dataset_release_id as pdf_release_id
from benchmarks.pdf_retrieval.solutions import solution_metadata as pdf_solution_metadata
from benchmarks.ranked_retrieval.definition import DATASET_ID as TEXT_DATASET_ID
from benchmarks.ranked_retrieval.definition import (
    DATASET_REVISION as TEXT_DATASET_REVISION,
)
from benchmarks.ranked_retrieval.definition import (
    build_task as build_text_task,
)
from benchmarks.ranked_retrieval.definition import dataset_release_id as text_release_id
from benchmarks.ranked_retrieval.solutions import solution_metadata as text_solution_metadata
from benchmarks.ranking_evidence import validate_artifact_scope, validate_language_slice
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


def _labels(rows: Dataset, *, id_field: str, fields: tuple[str, ...]) -> dict[str, dict[str, str]]:
    return {str(row[id_field]): {field: str(row[field]) for field in fields} for row in rows}


def _relevance(qrels: Dataset) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        if int(row["relevance"]) > 0:
            relevant[str(row["query_id"])].add(str(row["document_id"]))
    return dict(relevant)


def _exclusions(
    metadata: Mapping[str, Mapping[str, Any]], required: list[str], included: set[str]
) -> list[dict[str, Any]]:
    return [
        {
            "solution_id": solution_id,
            "reason": "unsupported-language"
            if not set(required) <= set(values["languages"])
            else "missing-scope-evidence",
        }
        for solution_id, values in sorted(metadata.items())
        if solution_id not in included
    ]


def _release_report(
    *,
    task_factory: Callable,
    release_id: Callable,
    dataset_repository: str,
    dataset_revision: str,
    root: Path,
    result_path: str,
    all_metadata: Mapping[str, Mapping[str, Any]],
    repository_url: str,
    commit: str,
    query_dimensions: tuple[Dimension, ...],
    document_dimensions: tuple[Dimension, ...],
    status: str = "historical-artifact",
) -> tuple[RunReport, list[dict[str, Any]], dict[str, Any]]:
    task = task_factory()
    dataset_id = release_id()
    dataset = task.datasets[dataset_id]
    coverage = {solution_id: values["languages"] for solution_id, values in all_metadata.items()}

    def load_scope(directory, scope_task, scope_id):
        artifact = RankingArtifact.load(directory)
        validate_artifact_scope(
            artifact.manifest,
            task=scope_task,
            dataset_id=scope_id,
            dataset_repository=dataset_repository,
            dataset_revision=dataset_revision,
        )
        scope_dataset = scope_task.datasets[scope_id]
        required = sorted(set(scope_dataset["queries"]["language"]))
        validate_solution_languages(list(artifact.rankings), coverage, required)
        return artifact

    artifact = load_scope(root, task, dataset_id)
    metadata = {solution_id: dict(all_metadata[solution_id]) for solution_id in artifact.rankings}
    result_url = f"{repository_url}/tree/{commit}/{result_path}"
    for values in metadata.values():
        values["result_url"] = result_url
    provenance = RunProvenance(
        source_commit=commit,
        source_module=".".join(part.replace("-", "_").replace(".", "_") for part in Path(result_path).parts),
        source_path=f"{result_path}/README.md",
        source_line=1,
        repository_url=repository_url,
        created_at=str(artifact.manifest["published_at"]),
    )
    report = artifact.evaluate(
        task=task,
        dataset_id=dataset_id,
        dataset=dataset,
        dataset_version=dataset_revision[:12],
        provenance=provenance,
        solution_metadata=metadata,
        status=status,
    )
    breakdowns = build_retrieval_breakdowns(
        task_id=task.id,
        dataset_id=dataset_id,
        dataset_version=dataset_revision[:12],
        rankings=artifact.rankings,
        relevant_by_query=_relevance(dataset["qrels"]),
        query_labels=_labels(
            dataset["queries"], id_field="query_id", fields=tuple(dimension.id for dimension in query_dimensions)
        ),
        document_labels=_labels(
            dataset["documents"],
            id_field="document_id",
            fields=tuple(dimension.id for dimension in document_dimensions),
        ),
        query_dimensions=query_dimensions,
        document_dimensions=document_dimensions,
    )
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def remember_evidence(current, languages, url):
        for solution_id in current.rankings:
            evidence[solution_id].append(
                {
                    "release": current.manifest["release"],
                    "languages": languages,
                    "queries": current.manifest["queries"],
                    "rankings_sha256": current.manifest["rankings_sha256"],
                    "source": current.manifest.get("source_runs", {}).get(solution_id, {}),
                    "result_url": url,
                }
            )

    remember_evidence(artifact, list(BENCHMARK_LANGUAGES), result_url)
    language_breakdown = next(item for item in breakdowns if item["id"] == "language")
    for language in BENCHMARK_LANGUAGES:
        directory = root / "slices" / language
        if not directory.exists():
            continue
        slice_id = release_id(language)
        slice_task = task_factory(language=language)
        slice_dataset = slice_task.datasets[slice_id]
        sliced = load_scope(directory, slice_task, slice_id)
        slice_task.validate_dataset(slice_dataset)
        validate_language_slice(dataset, slice_dataset, language)
        for rankings in sliced.rankings.values():
            slice_task.observations_from_rankings(slice_dataset, rankings, validate=False)
        value = next(value for value in language_breakdown["values"] if value["id"] == language)
        slice_breakdown = build_retrieval_breakdowns(
            task_id=task.id,
            dataset_id=dataset_id,
            dataset_version=dataset_revision[:12],
            rankings=sliced.rankings,
            relevant_by_query=_relevance(slice_dataset["qrels"]),
            query_labels=_labels(slice_dataset["queries"], id_field="query_id", fields=("language",)),
            document_labels={},
            query_dimensions=(Dimension("language", "Language", ((language, value["label"]),)),),
        )[0]["values"][0]
        for solution_id, rankings in sliced.rankings.items():
            if solution_id in artifact.rankings:
                full_slice = {query_id: artifact.rankings[solution_id][query_id] for query_id in rankings}
                if rankings != full_slice:
                    raise ValueError(f"Conflicting full and language evidence for {solution_id}")
            else:
                report.solutions[solution_id] = {
                    "id": solution_id,
                    **all_metadata[solution_id],
                    "result_url": f"{result_url}/slices/{language}",
                }
        value["results"].extend(
            row for row in slice_breakdown["results"] if row["solution_id"] not in artifact.rankings
        )
        remember_evidence(sliced, [language], f"{result_url}/slices/{language}")

    for breakdown in breakdowns:
        for value in breakdown["values"]:
            required = (
                [value["id"]]
                if breakdown["id"] in {"language", "relevant_document_language"}
                else list(BENCHMARK_LANGUAGES)
            )
            value["required_languages"] = required
            value["excluded_solutions"] = _exclusions(
                report.solutions, required, {row["solution_id"] for row in value["results"]}
            )
    scope_coverage = {
        "task_id": task.id,
        "dataset_id": dataset_id,
        "dataset_version": dataset_revision[:12],
        "required_languages": list(BENCHMARK_LANGUAGES),
        "solutions": [
            {
                "solution_id": solution_id,
                "evidence": evidence[solution_id],
            }
            for solution_id in sorted(report.solutions)
        ],
        "excluded_solutions": _exclusions(report.solutions, list(BENCHMARK_LANGUAGES), set(artifact.rankings)),
    }
    return report, breakdowns, scope_coverage


def _text_release(repository_url: str, commit: str):
    return _release_report(
        task_factory=build_text_task,
        release_id=text_release_id,
        dataset_repository=TEXT_DATASET_ID,
        dataset_revision=TEXT_DATASET_REVISION,
        root=TEXT_RESULT_ROOT,
        result_path="results/modern-ir-v2.2",
        all_metadata=text_solution_metadata(repository_url, commit),
        repository_url=repository_url,
        commit=commit,
        query_dimensions=TEXT_QUERY_DIMENSIONS,
        document_dimensions=TEXT_DOCUMENT_DIMENSIONS,
    )


def _pdf_release(repository_url: str, commit: str):
    return _release_report(
        task_factory=build_pdf_task,
        release_id=pdf_release_id,
        dataset_repository=PDF_DATASET_ID,
        dataset_revision=PDF_DATASET_REVISION,
        root=PDF_RESULT_ROOT,
        result_path="results/modern-ir-pdf-v1",
        all_metadata=pdf_solution_metadata(repository_url, commit),
        repository_url=repository_url,
        commit=commit,
        query_dimensions=PDF_QUERY_DIMENSIONS,
        document_dimensions=PDF_DOCUMENT_DIMENSIONS,
        status="live-artifact",
    )


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
    text_report, text_breakdowns, text_coverage = _text_release(repository_url, commit)
    pdf_report, pdf_breakdowns, pdf_coverage = _pdf_release(repository_url, commit)
    text_report.extend(pdf_report)
    return build_space_payload(
        text_report,
        release="modern-ir-bench-public-20261011",
        counts={"documents": 5000, "queries": 1000, "qrels": 6575},
        languages=("Chinese", "English"),
        breakdowns=(*text_breakdowns, *pdf_breakdowns),
        coverage=(text_coverage, pdf_coverage),
        notice=(
            "Scores are rebuilt from versioned result evidence. Open a Solution for its "
            "method notes, code, and provenance."
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
