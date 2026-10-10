"""Publish the reviewed historical v2.2 ranking artifact to the Space."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from benchmarks.modern_ir_breakdowns import build_breakdowns
from benchmarks.modern_ir_definition import (
    DATASET_ID,
    DATASET_REVISION,
    RELEASE_ID,
    build_task,
)
from benchmarks.modern_ir_solutions import solution_metadata
from modern_ir_bench import RunProvenance
from modern_ir_bench.exporters import write_space_results
from modern_ir_bench.results import RankingArtifact

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "results/modern-ir-v2.2"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/modern-ir-retrieval"
SPACE_RESULTS = PROJECT_ROOT / "space/data/results.json"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository_url(remote: str) -> str:
    if remote.startswith("git@github.com:"):
        remote = f"https://github.com/{remote.removeprefix('git@github.com:')}"
    return remote.removesuffix(".git")


def main() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("Commit all source changes before publishing reproducible Space metadata")
    commit = _git("rev-parse", "HEAD")
    repository_url = _repository_url(_git("remote", "get-url", "origin"))
    artifact = RankingArtifact.load(ARTIFACT_ROOT)
    task = build_task()
    dataset = task.datasets[RELEASE_ID]
    provenance = RunProvenance(
        source_commit=commit,
        source_module="results.modern_ir_v2_2",
        source_path="results/modern-ir-v2.2/README.md",
        source_line=1,
        repository_url=repository_url,
        created_at=str(
            artifact.manifest.get(
                "published_at",
                datetime.now(timezone.utc).isoformat(),
            )
        ),
    )
    metadata = solution_metadata(repository_url, commit)
    evidence_url = provenance.source_url
    for values in metadata.values():
        values["result_url"] = evidence_url
    report = artifact.evaluate(
        task=task,
        dataset_id=RELEASE_ID,
        dataset=dataset,
        dataset_version=DATASET_REVISION[:12],
        provenance=provenance,
        solution_metadata=metadata,
    )

    documents = list(load_dataset(DATASET_ID, "corpus", split="corpus", revision=DATASET_REVISION))
    queries = list(load_dataset(DATASET_ID, "queries", split="queries", revision=DATASET_REVISION))
    qrels = list(load_dataset(DATASET_ID, "qrels", split="qrels", revision=DATASET_REVISION))
    breakdowns = build_breakdowns(
        documents=documents,
        queries=queries,
        qrels=qrels,
        artifact=artifact,
        report=report,
    )
    payload = {
        "release": RELEASE_ID,
        "artifact": artifact.manifest,
        "results": report.records,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in (OUTPUT_ROOT / "space-results.json", SPACE_RESULTS):
        write_space_results(
            report,
            path,
            release=RELEASE_ID,
            counts={"documents": 5000, "queries": 1000, "qrels": 6575},
            languages=("Chinese", "English"),
            breakdowns=breakdowns,
            notice=(
                "v2.2 scores are recomputed from the reviewed historical Top-10 ranking "
                "artifact. Solution links open the maintained executable definitions; "
                "result evidence is linked separately."
            ),
        )
    report.write_observations(OUTPUT_ROOT / "observations")


if __name__ == "__main__":
    main()
