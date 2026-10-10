"""Run all executable Solutions on the current Modern IR benchmark."""

from __future__ import annotations

import gzip
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.modern_ir_definition import (
    DATASET_ID,
    DATASET_REVISION,
    RELEASE_ID,
    build_task,
)
from benchmarks.modern_ir_solutions import build_solutions, solution_metadata
from modern_ir_bench import RunProvenance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/modern-ir-live"


def _write_rankings(report, path: Path, *, top_k: int) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for observation_key, observations in report.observations.items():
            solution_id = observation_key.rsplit("__", 1)[-1]
            for row in observations:
                if len(row["ranked_ids"]) != top_k:
                    raise RuntimeError(
                        f"{solution_id} returned {len(row['ranked_ids'])} hits for {row['query_id']}; expected {top_k}"
                    )
                stream.write(
                    json.dumps(
                        {
                            "query_id": row["query_id"],
                            "solution_id": solution_id,
                            "ranked_doc_ids": row["ranked_ids"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows += 1
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    task = build_task()
    provenance = RunProvenance.capture(
        source_module="benchmarks.modern_ir",
        source_path="benchmarks/modern_ir.py",
        source_line=inspect.getsourcelines(main)[1],
    )
    report = task.run(
        build_solutions(),
        provenance=provenance,
        status="live",
    )
    metadata = solution_metadata(provenance.repository_url, provenance.source_commit)
    for solution_id, values in metadata.items():
        report.solutions[solution_id].update(values)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "results.json").write_text(
        json.dumps(
            {
                "release": RELEASE_ID,
                "results": report.records,
                "solutions": list(report.solutions.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ranking_path = OUTPUT_ROOT / "rankings.jsonl.gz"
    ranking_rows, ranking_checksum = _write_rankings(
        report,
        ranking_path,
        top_k=task.top_k,
    )
    query_count = len({row["query_id"] for observations in report.observations.values() for row in observations})
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "live-top-k-rankings",
                "release": RELEASE_ID,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_commit": provenance.source_commit,
                "source_url": provenance.source_url,
                "dataset": {
                    "repository": DATASET_ID,
                    "revision": DATASET_REVISION,
                },
                "solutions": sorted(report.solutions),
                "queries": query_count,
                "ranking_rows": ranking_rows,
                "top_k": task.top_k,
                "rankings_sha256": ranking_checksum,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report.write_observations(OUTPUT_ROOT / "observations")


if __name__ == "__main__":
    main()
