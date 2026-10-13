"""Compare score fusion against rank fusion on the PDF retrieval release."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
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
        "paddleocr-v5-mobile-bm25-bge-m3-weighted-25-75",
        "OCR BM25 + BGE-M3",
    ),
    (
        "paddleocr-v5-mobile-bm25-voyage-multimodal-3-5-rrf",
        "paddleocr-v5-mobile-bm25-voyage-multimodal-3-5-weighted-25-75",
        "OCR BM25 + Voyage Multimodal 3.5",
    ),
)


def _relevant_documents(qrels: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        if int(row["relevance"]) > 0:
            relevant[str(row["query_id"])].add(str(row["document_id"]))
    return dict(relevant)


def _comparison(
    *,
    rrf_id: str,
    weighted_id: str,
    label: str,
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    relevant: Mapping[str, set[str]],
    languages: Mapping[str, str],
) -> dict[str, Any]:
    rrf = {
        query_id: retrieval_metrics(relevant[query_id], ranked)
        for query_id, ranked in rankings[rrf_id].items()
    }
    weighted = {
        query_id: retrieval_metrics(relevant[query_id], ranked)
        for query_id, ranked in rankings[weighted_id].items()
    }
    deltas = {query_id: weighted[query_id]["ndcg@10"] - rrf[query_id]["ndcg@10"] for query_id in relevant}
    outcomes = Counter(
        "improved" if delta > 1e-12 else "regressed" if delta < -1e-12 else "unchanged"
        for delta in deltas.values()
    )
    return {
        "label": label,
        "rrf_solution": rrf_id,
        "weighted_solution": weighted_id,
        "rrf_ndcg_at_10": round(statistics.mean(row["ndcg@10"] for row in rrf.values()), 8),
        "weighted_ndcg_at_10": round(statistics.mean(row["ndcg@10"] for row in weighted.values()), 8),
        "weighted_delta": round(statistics.mean(deltas.values()), 8),
        "outcomes": dict(sorted(outcomes.items())),
        "language_deltas": {
            language: round(
                statistics.mean(delta for query_id, delta in deltas.items() if languages[query_id] == language),
                8,
            )
            for language in sorted(set(languages.values()))
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Modern IR PDF Retrieval v1 fusion diagnostics",
        "",
        "The fixed 25/75 score weights are compared with equal-weight reciprocal rank fusion on the same Top-50 branches.",
        "",
        "| Retrieval branches | RRF NDCG@10 | Weighted NDCG@10 | Weighted delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["comparisons"]:
        lines.append(
            f"| {row['label']} | {row['rrf_ndcg_at_10']:.4f} | {row['weighted_ndcg_at_10']:.4f} | "
            f"{row['weighted_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "The text benchmark's 25/75 lexical/dense weights do not transfer to PDF retrieval. "
                "Per-query min-max normalization does not make OCR lexical scores and visual or dense scores reliably "
                "comparable. RRF remains the recommended default for these heterogeneous PDF branches."
            ),
            "",
            (
                "The weighted runs remain published as negative evidence. Detailed outcome counts and language slices "
                "are stored in `analysis.json`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_report(result_root: Path) -> dict[str, Any]:
    artifact = RankingArtifact.load(result_root)
    dataset = build_task().datasets[RELEASE_ID]
    relevant = _relevant_documents(dataset["qrels"])
    languages = {str(row["query_id"]): str(row["language"]) for row in dataset["queries"]}
    report = {
        "schema_version": 1,
        "release": RELEASE_ID,
        "ranking_evidence_sha256": artifact.manifest["rankings_sha256"],
        "comparisons": [
            _comparison(
                rrf_id=rrf_id,
                weighted_id=weighted_id,
                label=label,
                rankings=artifact.rankings,
                relevant=relevant,
                languages=languages,
            )
            for rrf_id, weighted_id, label in COMPARISONS
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
