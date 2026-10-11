"""Run the maintained PDF Retrieval Solutions on the public PDF release."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from pathlib import Path

from benchmarks.pdf_retrieval.definition import RELEASE_ID, build_task
from benchmarks.pdf_retrieval.solutions import SOLUTION_SPECS, build_solutions
from modern_ir_bench import RunProvenance
from modern_ir_bench.retrieval.indexes.milvus import MilvusLite, MilvusServer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/modern-ir-pdf-live"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PDF Retrieval Solutions")
    parser.add_argument(
        "--solution",
        action="append",
        choices=[spec.id for spec in SOLUTION_SPECS],
        help="Solution to run; repeat the flag to select several. Defaults to the complete catalog.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    task = build_task()
    lite_path = os.environ.get("MIR_MILVUS_LITE_PATH")
    target = (
        MilvusLite(lite_path)
        if lite_path
        else MilvusServer(
            os.environ.get("MIR_MILVUS_URI", "http://127.0.0.1:19530"),
            token=os.environ.get("MIR_MILVUS_TOKEN", ""),
        )
    )
    solutions = build_solutions(
        target=target,
        project_root=PROJECT_ROOT,
        solution_ids=arguments.solution,
    )
    provenance = RunProvenance.capture(
        source_module="benchmarks.pdf_retrieval.run",
        source_path="benchmarks/pdf_retrieval/run.py",
        source_line=inspect.getsourcelines(main)[1],
    )
    report = task.run(solutions, provenance=provenance, status="live")
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
    report.write_observations(OUTPUT_ROOT / "observations")


if __name__ == "__main__":
    main()
