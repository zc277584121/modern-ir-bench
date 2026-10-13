"""Analyze stored retrieval candidates without running retrieval again."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = PROJECT_ROOT / "space/data/fusion"
DEFAULT_JSON = PROJECT_ROOT / "results/candidate-pools/analysis.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "results/candidate-pools/README.md"


@dataclass(frozen=True)
class CandidateGroup:
    """A named union of stored retrieval routes."""

    id: str
    label: str
    routes: tuple[str, ...]


GROUPS = {
    "modern-ir-ranked-retrieval-v2.2": (
        CandidateGroup("voyage-bm25", "Voyage 4 Large + BM25", ("voyage-4-large", "bm25")),
        CandidateGroup(
            "voyage-gte-sparse",
            "Voyage 4 Large + GTE sparse",
            ("voyage-4-large", "gte-multilingual-sparse"),
        ),
        CandidateGroup(
            "voyage-gte-sparse-bm25",
            "Voyage 4 Large + GTE sparse + BM25",
            ("voyage-4-large", "gte-multilingual-sparse", "bm25"),
        ),
    ),
    "modern-ir-pdf-retrieval-v1": (
        CandidateGroup(
            "ocr-voyage-visual",
            "OCR Voyage 4 Large + Voyage Multimodal 3.5",
            ("ocr-voyage-4-large", "voyage-multimodal-3-5"),
        ),
        CandidateGroup(
            "ocr-voyage-bm25",
            "OCR Voyage 4 Large + OCR BM25",
            ("ocr-voyage-4-large", "ocr-bm25"),
        ),
        CandidateGroup(
            "ocr-voyage-visual-bm25",
            "OCR Voyage 4 Large + Voyage Multimodal 3.5 + OCR BM25",
            ("ocr-voyage-4-large", "voyage-multimodal-3-5", "ocr-bm25"),
        ),
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _unpack_ranking(packed: Sequence[float | int]) -> tuple[int, ...]:
    return tuple(int(packed[index]) for index in range(0, len(packed), 2))


def _oracle_ndcg_at_10(found_relevant: int, total_relevant: int) -> float:
    """Return the best possible NDCG@10 using only the supplied candidates."""

    dcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(found_relevant, 10) + 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(total_relevant, 10) + 1))
    return dcg / ideal


def _summarize_candidates(
    candidate_sets: Sequence[set[int]],
    relevant_sets: Sequence[set[int]],
) -> dict[str, float | int]:
    if len(candidate_sets) != len(relevant_sets) or not candidate_sets:
        raise ValueError("Candidate and relevance rows must have the same non-zero length")
    found = [len(candidates.intersection(relevant)) for candidates, relevant in zip(candidate_sets, relevant_sets, strict=True)]
    return {
        "queries": len(relevant_sets),
        "mean_candidates": round(statistics.mean(map(len, candidate_sets)), 4),
        "relevant_coverage": round(
            statistics.mean(count / len(relevant) for count, relevant in zip(found, relevant_sets, strict=True)),
            8,
        ),
        "queries_with_any_relevant": round(statistics.mean(count > 0 for count in found), 8),
        "queries_with_all_relevant": round(
            statistics.mean(count == len(relevant) for count, relevant in zip(found, relevant_sets, strict=True)),
            8,
        ),
        "oracle_ndcg_at_10": round(
            statistics.mean(
                _oracle_ndcg_at_10(count, len(relevant))
                for count, relevant in zip(found, relevant_sets, strict=True)
            ),
            8,
        ),
    }


def analyze_benchmark(
    benchmark: Mapping[str, Any],
    route_rankings: Mapping[str, Sequence[Sequence[int]]],
    groups: Sequence[CandidateGroup],
) -> dict[str, Any]:
    """Measure coverage, overlap, and complementarity for one benchmark."""

    relevant_sets = [set(map(int, row["relevant"])) for row in benchmark["queries"]]
    route_sets = {
        route_id: [set(map(int, ranking)) for ranking in rankings]
        for route_id, rankings in route_rankings.items()
    }
    if any(len(rankings) != len(relevant_sets) for rankings in route_sets.values()):
        raise ValueError("Every route must cover every query")

    route_labels = {str(route["id"]): str(route["label"]) for route in benchmark["routes"]}
    route_rows = []
    all_route_ids = tuple(route_sets)
    for route_id, candidate_sets in route_sets.items():
        other_routes = tuple(candidate for candidate in all_route_ids if candidate != route_id)
        unique_relevant = 0
        for index, relevant in enumerate(relevant_sets):
            other_relevant = set().union(*(route_sets[other][index].intersection(relevant) for other in other_routes))
            unique_relevant += len(candidate_sets[index].intersection(relevant) - other_relevant)
        route_rows.append(
            {
                "id": route_id,
                "label": route_labels[route_id],
                **_summarize_candidates(candidate_sets, relevant_sets),
                "globally_unique_relevant": unique_relevant,
            }
        )

    group_rows = []
    configured_groups = tuple(groups) + (
        CandidateGroup("all-routes", "All stored routes", all_route_ids),
    )
    for group in configured_groups:
        missing = set(group.routes) - set(route_sets)
        if missing:
            raise ValueError(f"Candidate group {group.id!r} references unknown routes: {sorted(missing)}")
        unions = [
            set().union(*(route_sets[route_id][index] for route_id in group.routes))
            for index in range(len(relevant_sets))
        ]
        group_rows.append(
            {
                "id": group.id,
                "label": group.label,
                "routes": list(group.routes),
                **_summarize_candidates(unions, relevant_sets),
            }
        )

    pair_rows = []
    for left_index, left_id in enumerate(all_route_ids):
        for right_id in all_route_ids[left_index + 1 :]:
            left = route_sets[left_id]
            right = route_sets[right_id]
            jaccards = [
                len(left_row.intersection(right_row)) / len(left_row.union(right_row))
                for left_row, right_row in zip(left, right, strict=True)
            ]
            left_unique = sum(
                len((left_row - right_row).intersection(relevant))
                for left_row, right_row, relevant in zip(left, right, relevant_sets, strict=True)
            )
            right_unique = sum(
                len((right_row - left_row).intersection(relevant))
                for left_row, right_row, relevant in zip(left, right, relevant_sets, strict=True)
            )
            pair_rows.append(
                {
                    "left": left_id,
                    "right": right_id,
                    "mean_jaccard": round(statistics.mean(jaccards), 8),
                    "left_unique_relevant": left_unique,
                    "right_unique_relevant": right_unique,
                }
            )

    return {
        "task_id": benchmark["task_id"],
        "dataset_id": benchmark["dataset_id"],
        "dataset_version": benchmark["dataset_version"],
        "depth": benchmark["max_depth"],
        "queries": len(relevant_sets),
        "positive_qrels": sum(map(len, relevant_sets)),
        "routes": route_rows,
        "groups": group_rows,
        "pairs": pair_rows,
    }


def build_report(bundle_root: Path) -> dict[str, Any]:
    """Load one Fusion Lab bundle and analyze every published task."""

    manifest = _read_json(bundle_root / "manifest.json.gz")
    analyses = []
    for benchmark in manifest["benchmarks"]:
        route_rankings = {}
        for route in benchmark["routes"]:
            payload = _read_json(bundle_root / route["path"])
            route_rankings[str(route["id"])] = [
                _unpack_ranking(ranking) for ranking in payload["rankings"]
            ]
        analyses.append(
            analyze_benchmark(
                benchmark,
                route_rankings,
                GROUPS.get(str(benchmark["task_id"]), ()),
            )
        )
    return {"schema_version": 1, "benchmarks": analyses}


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the compact human-readable report."""

    lines = [
        "# Candidate pool diagnostics",
        "",
        "This report uses stored Top-50 route evidence. It does not run embedding, retrieval, or reranking.",
        "Recall is averaged per query. Oracle NDCG@10 assumes a perfect reranker over the candidate set.",
    ]
    for benchmark in report["benchmarks"]:
        lines.extend(
            [
                "",
                f"## {benchmark['task_id']}",
                "",
                f"{benchmark['queries']:,} queries · {benchmark['positive_qrels']:,} positive Qrels",
                "",
                "### Atomic routes",
                "",
                "| Route | Recall@50 | Oracle NDCG@10 | All positives found | Unique positives |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in benchmark["routes"]:
            lines.append(
                f"| {row['label']} | {row['relevant_coverage']:.4f} | {row['oracle_ndcg_at_10']:.4f} | "
                f"{row['queries_with_all_relevant']:.1%} | {row['globally_unique_relevant']} |"
            )
        lines.extend(
            [
                "",
                "### Candidate unions",
                "",
                "| Routes | Mean candidates | Relevant coverage | Oracle NDCG@10 | All positives found |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in benchmark["groups"]:
            lines.append(
                f"| {row['label']} | {row['mean_candidates']:.1f} | {row['relevant_coverage']:.4f} | "
                f"{row['oracle_ndcg_at_10']:.4f} | {row['queries_with_all_relevant']:.1%} |"
            )
        lines.extend(
            [
                "",
                "### Pairwise complementarity",
                "",
                "| Left | Right | Candidate Jaccard | Relevant only in left | Relevant only in right |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        labels = {row["id"]: row["label"] for row in benchmark["routes"]}
        for row in benchmark["pairs"]:
            lines.append(
                f"| {labels[row['left']]} | {labels[row['right']]} | {row['mean_jaccard']:.3f} | "
                f"{row['left_unique_relevant']} | {row['right_unique_relevant']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    arguments = parser.parse_args()
    report = build_report(arguments.bundle.resolve())
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    arguments.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
