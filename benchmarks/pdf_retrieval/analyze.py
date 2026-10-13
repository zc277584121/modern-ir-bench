"""Summarize published PDF fusion Solutions against their component baselines."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.pdf_retrieval.definition import RELEASE_ID, build_task
from benchmarks.reporting import retrieval_metrics
from modern_ir_bench.results import RankingArtifact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results/modern-ir-pdf-v1"
COMPARISONS = (
    (
        "paddleocr-v5-mobile-bm25-bge-m3-rrf",
        ("paddleocr-v5-mobile-bm25-full", "paddleocr-v5-mobile-bge-m3-full"),
        "OCR BM25 + BGE-M3",
    ),
    (
        "paddleocr-v5-mobile-bm25-voyage-multimodal-3-5-rrf",
        ("paddleocr-v5-mobile-bm25-full", "voyage-multimodal-3-5-pages"),
        "OCR BM25 + Voyage Multimodal 3.5",
    ),
    (
        "paddleocr-v5-mobile-voyage-4-large-voyage-multimodal-3-5-rrf",
        ("paddleocr-v5-mobile-voyage-4-large-full", "voyage-multimodal-3-5-pages"),
        "OCR Voyage 4 Large + Voyage Multimodal 3.5",
    ),
)


def _relevant_documents(qrels: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        if int(row["relevance"]) > 0:
            relevant[str(row["query_id"])].add(str(row["document_id"]))
    return dict(relevant)


def _ndcg(
    rankings: Mapping[str, Sequence[str]],
    relevant: Mapping[str, set[str]],
) -> float:
    return statistics.mean(
        retrieval_metrics(relevant[query_id], ranking)["ndcg@10"]
        for query_id, ranking in rankings.items()
    )


def _comparison(
    *,
    fusion_id: str,
    component_ids: tuple[str, str],
    label: str,
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    relevant: Mapping[str, set[str]],
) -> dict[str, Any]:
    fusion_score = _ndcg(rankings[fusion_id], relevant)
    component_scores = {
        component_id: _ndcg(rankings[component_id], relevant)
        for component_id in component_ids
    }
    best_component = max(component_scores.values())
    return {
        "label": label,
        "fusion_solution": fusion_id,
        "component_solutions": list(component_ids),
        "fusion_ndcg_at_10": round(fusion_score, 8),
        "component_ndcg_at_10": {
            solution_id: round(score, 8) for solution_id, score in component_scores.items()
        },
        "delta_vs_best_component": round(fusion_score - best_component, 8),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Modern IR PDF Retrieval v1 fusion diagnostics",
        "",
        "Published RRF Solutions are compared with their independently published component baselines.",
        "All fusion routes contribute Top-50 candidates and use equal-weight RRF with `k=60`.",
        "",
        "| Retrieval routes | Fusion NDCG@10 | Best component NDCG@10 | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["comparisons"]:
        best_component = max(row["component_ndcg_at_10"].values())
        lines.append(
            f"| {row['label']} | {row['fusion_ndcg_at_10']:.4f} | {best_component:.4f} | "
            f"{row['delta_vs_best_component']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Alternative route weights remain exploratory and can be computed from stored Top-50 evidence in Fusion Lab.",
            "They are not published as separate leaderboard Solutions.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(result_root: Path) -> dict[str, Any]:
    artifact = RankingArtifact.load(result_root)
    dataset = build_task().datasets[RELEASE_ID]
    relevant = _relevant_documents(dataset["qrels"])
    report = {
        "schema_version": 1,
        "release": RELEASE_ID,
        "ranking_evidence_sha256": artifact.manifest["rankings_sha256"],
        "comparisons": [
            _comparison(
                fusion_id=fusion_id,
                component_ids=component_ids,
                label=label,
                rankings=artifact.rankings,
                relevant=relevant,
            )
            for fusion_id, component_ids, label in COMPARISONS
        ],
    }
    (result_root / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (result_root / "DIAGNOSTICS.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    arguments = parser.parse_args()
    report = build_report(arguments.result_root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
