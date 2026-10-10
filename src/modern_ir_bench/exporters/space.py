"""Export a run report for the read-only Hugging Face Space."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from modern_ir_bench.core.result import RunReport


def build_space_payload(
    report: RunReport,
    *,
    release: str,
    counts: Mapping[str, int] | None = None,
    languages: Sequence[str] = (),
    breakdowns: Sequence[Mapping[str, object]] = (),
    notice: str | None = None,
) -> dict[str, object]:
    benchmark: dict[str, object] = {
        "title": "Modern IR Bench",
        "release": release,
    }
    if counts:
        benchmark["counts"] = dict(counts)
    if languages:
        benchmark["languages"] = list(languages)
    if notice:
        benchmark["notice"] = notice
    payload: dict[str, object] = {
        "benchmark": benchmark,
        "tasks": sorted(report.tasks.values(), key=lambda item: item["id"]),
        "solutions": sorted(report.solutions.values(), key=lambda item: item["id"]),
        "results": report.records,
    }
    if breakdowns:
        payload["breakdowns"] = list(breakdowns)
    return payload


def write_space_results(
    report: RunReport,
    path: Path,
    *,
    release: str,
    counts: Mapping[str, int] | None = None,
    languages: Sequence[str] = (),
    breakdowns: Sequence[Mapping[str, object]] = (),
    notice: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_space_payload(
                report,
                release=release,
                counts=counts,
                languages=languages,
                breakdowns=breakdowns,
                notice=notice,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
