"""Run the maintained PDF Retrieval Solutions on the public PDF release."""

from __future__ import annotations

import inspect
import json
import logging
import os
from pathlib import Path

from benchmarks.pdf_retrieval.definition import RELEASE_ID, build_task
from benchmarks.pdf_retrieval.solution import build_solution
from modern_ir_bench import RunProvenance
from modern_ir_bench.retrieval.indexes.milvus import MilvusServer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/modern-ir-pdf-live"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    task = build_task()
    target = MilvusServer(
        os.environ.get("MIR_MILVUS_URI", "http://127.0.0.1:19530"),
        token=os.environ.get("MIR_MILVUS_TOKEN", ""),
    )
    solution = build_solution(
        target,
        device=os.environ.get("MIR_OCR_DEVICE", "cpu"),
        cache_dir=PROJECT_ROOT / ".runtime/pdf-ocr/pp-ocrv5-mobile",
    )
    provenance = RunProvenance.capture(
        source_module="benchmarks.pdf_retrieval.run",
        source_path="benchmarks/pdf_retrieval/run.py",
        source_line=inspect.getsourcelines(main)[1],
    )
    report = task.run([solution], provenance=provenance, status="live")
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
