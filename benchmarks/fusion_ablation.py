"""Compare dense, lexical, and learned-sparse retrieval with stored Top-50 evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.reporting import METRIC_ORDER, retrieval_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = PROJECT_ROOT / "space/data/fusion"
DEFAULT_JSON = PROJECT_ROOT / "results/candidate-pools/fusion-ablation.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "docs/experiments/dense-lexical-sparse-ablation.md"
TASK_ID = "modern-ir-ranked-retrieval-v2.2"
DENSE_ROUTE = "voyage-4-large"
LEXICAL_ROUTE = "bm25-tuned"
BASE_SPARSE_ROUTES = ("bge-m3-sparse", "gte-multilingual-sparse")
OPTIONAL_SPARSE_SCOPES = {"splade-v3-full": "en"}
FOLDS = 5
WEIGHT_STEP = 0.05

PackedRanking = Sequence[float | int]
Ranking = tuple[int, ...]


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _load_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, list[PackedRanking]]]:
    manifest = _read_gzip_json(bundle / "manifest.json.gz")
    benchmark = next(row for row in manifest["benchmarks"] if row["task_id"] == TASK_ID)
    routes = {route["id"]: _read_gzip_json(bundle / route["path"])["rankings"] for route in benchmark["routes"]}
    required = {DENSE_ROUTE, LEXICAL_ROUTE, *BASE_SPARSE_ROUTES}
    if missing := required - set(routes):
        raise ValueError(f"Fusion bundle is missing required routes: {sorted(missing)}")
    return benchmark, routes


def _hits(packed: PackedRanking, depth: int = 50) -> tuple[tuple[int, float], ...]:
    limit = min(len(packed), depth * 2)
    return tuple((int(packed[index]), float(packed[index + 1])) for index in range(0, limit, 2))


def _normalize(packed: PackedRanking) -> dict[int, float]:
    hits = _hits(packed)
    values = [score for _, score in hits]
    low = min(values)
    high = max(values)
    if high == low:
        return {document_id: 0.0 for document_id, _ in hits}
    return {document_id: (score - low) / (high - low) for document_id, score in hits}


def _weighted_ranking(
    normalized: Mapping[str, Sequence[Mapping[int, float]]],
    route_ids: Sequence[str],
    weights: Sequence[float],
    query_index: int,
) -> Ranking:
    scores: dict[int, float] = {}
    for route_id, weight in zip(route_ids, weights, strict=True):
        if weight == 0:
            continue
        for document_id, score in normalized[route_id][query_index].items():
            scores[document_id] = scores.get(document_id, 0.0) + weight * score
    return tuple(document_id for document_id, _ in sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:10])


def _rrf_ranking(
    routes: Mapping[str, Sequence[PackedRanking]],
    route_ids: Sequence[str],
    query_index: int,
    rank_constant: int = 60,
) -> Ranking:
    scores: dict[int, float] = {}
    for route_id in route_ids:
        for rank, (document_id, _) in enumerate(_hits(routes[route_id][query_index]), start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1 / (rank_constant + rank)
    return tuple(document_id for document_id, _ in sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:10])


def _weight_grid(size: int) -> tuple[tuple[float, ...], ...]:
    units = round(1 / WEIGHT_STEP)

    def compositions(remaining: int, slots: int) -> list[tuple[int, ...]]:
        if slots == 1:
            return [(remaining,)]
        return [(value, *tail) for value in range(remaining + 1) for tail in compositions(remaining - value, slots - 1)]

    return tuple(tuple(value / units for value in row) for row in compositions(units, size))


def _fold(query_id: str) -> int:
    return int(hashlib.sha256(query_id.encode()).hexdigest()[:8], 16) % FOLDS


def _average_metrics(relevant: Sequence[set[int]], rankings: Sequence[Ranking]) -> dict[str, float]:
    per_query = [
        retrieval_metrics({str(value) for value in expected}, [str(value) for value in ranking])
        for expected, ranking in zip(relevant, rankings, strict=True)
    ]
    return {metric: round(statistics.mean(row[metric] for row in per_query), 8) for metric in METRIC_ORDER}


def _language_metrics(
    relevant: Sequence[set[int]],
    rankings: Sequence[Ranking],
    languages: Sequence[str],
) -> dict[str, dict[str, float]]:
    return {
        language: _average_metrics(
            [expected for expected, value in zip(relevant, languages, strict=True) if value == language],
            [ranking for ranking, value in zip(rankings, languages, strict=True) if value == language],
        )
        for language in sorted(set(languages))
    }


def _ndcg(expected: set[int], ranking: Ranking) -> float:
    gains = [document_id in expected for document_id in ranking[:10]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(expected), 10) + 1))
    return dcg / ideal


def _weighted_cross_validation(
    *,
    query_ids: Sequence[str],
    relevant: Sequence[set[int]],
    languages: Sequence[str],
    normalized: Mapping[str, Sequence[Mapping[int, float]]],
    route_ids: tuple[str, ...],
) -> dict[str, Any]:
    grid = _weight_grid(len(route_ids))
    ndcg_by_weights: dict[tuple[float, ...], tuple[float, ...]] = {}
    for weights in grid:
        ndcg_by_weights[weights] = tuple(
            _ndcg(expected, _weighted_ranking(normalized, route_ids, weights, index))
            for index, expected in enumerate(relevant)
        )

    folds = tuple(_fold(query_id) for query_id in query_ids)
    selected_weights: list[tuple[float, ...]] = []
    oof_rankings: list[Ranking | None] = [None] * len(query_ids)
    fold_ndcg: list[float] = []
    for held_out in range(FOLDS):
        training = [index for index, fold in enumerate(folds) if fold != held_out]
        validation = [index for index, fold in enumerate(folds) if fold == held_out]
        best = max(
            grid,
            key=lambda weights: (
                statistics.mean(ndcg_by_weights[weights][index] for index in training),
                min(weights),
                weights,
            ),
        )
        selected_weights.append(best)
        for index in validation:
            oof_rankings[index] = _weighted_ranking(normalized, route_ids, best, index)
        fold_ndcg.append(round(statistics.mean(ndcg_by_weights[best][index] for index in validation), 8))

    full_best = max(
        grid,
        key=lambda weights: (statistics.mean(ndcg_by_weights[weights]), min(weights), weights),
    )
    rankings = [ranking for ranking in oof_rankings if ranking is not None]
    if len(rankings) != len(query_ids):
        raise RuntimeError("Cross-validation did not produce one out-of-fold ranking per query")
    return {
        "routes": list(route_ids),
        "method": "weighted-min-max",
        "weight_step": WEIGHT_STEP,
        "full_best_weights": list(full_best),
        "fold_weights": [list(weights) for weights in selected_weights],
        "fold_ndcg_at_10": fold_ndcg,
        "metrics": _average_metrics(relevant, rankings),
        "language_metrics": _language_metrics(relevant, rankings, languages),
    }


def _atomic_result(
    route_id: str,
    routes: Mapping[str, Sequence[PackedRanking]],
    relevant: Sequence[set[int]],
    languages: Sequence[str],
) -> dict[str, Any]:
    rankings = [tuple(document_id for document_id, _ in _hits(row, depth=10)) for row in routes[route_id]]
    return {
        "routes": [route_id],
        "method": "atomic",
        "metrics": _average_metrics(relevant, rankings),
        "language_metrics": _language_metrics(relevant, rankings, languages),
    }


def _rrf_result(
    route_ids: tuple[str, ...],
    routes: Mapping[str, Sequence[PackedRanking]],
    relevant: Sequence[set[int]],
    languages: Sequence[str],
) -> dict[str, Any]:
    rankings = [_rrf_ranking(routes, route_ids, index) for index in range(len(relevant))]
    return {
        "routes": list(route_ids),
        "method": "rrf",
        "rank_constant": 60,
        "metrics": _average_metrics(relevant, rankings),
        "language_metrics": _language_metrics(relevant, rankings, languages),
    }


def build_report(bundle: Path) -> dict[str, Any]:
    benchmark, routes = _load_bundle(bundle)
    query_ids = [str(row["id"]) for row in benchmark["queries"]]
    relevant = [set(map(int, row["relevant"])) for row in benchmark["queries"]]
    languages = [str(row["labels"][0]) for row in benchmark["queries"]]
    sparse_routes = (*BASE_SPARSE_ROUTES, *(route for route in OPTIONAL_SPARSE_SCOPES if route in routes))
    normalized = {
        route_id: [_normalize(row) for row in rankings]
        for route_id, rankings in routes.items()
        if route_id in {DENSE_ROUTE, LEXICAL_ROUTE, *sparse_routes}
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "task_id": benchmark["task_id"],
        "dataset_id": benchmark["dataset_id"],
        "dataset_version": benchmark["dataset_version"],
        "candidate_depth": 50,
        "queries": len(query_ids),
        "folds": FOLDS,
        "dense": _atomic_result(DENSE_ROUTE, routes, relevant, languages),
        "studies": [],
    }
    for sparse_route in sparse_routes:
        scope = OPTIONAL_SPARSE_SCOPES.get(sparse_route)
        indices = [index for index, language in enumerate(languages) if scope is None or language == scope]
        scoped_query_ids = [query_ids[index] for index in indices]
        scoped_relevant = [relevant[index] for index in indices]
        scoped_languages = [languages[index] for index in indices]
        scoped_routes = {route_id: [rankings[index] for index in indices] for route_id, rankings in routes.items()}
        scoped_normalized = {
            route_id: [rankings[index] for index in indices] for route_id, rankings in normalized.items()
        }
        configurations = {
            "dense+bm25": (DENSE_ROUTE, LEXICAL_ROUTE),
            "dense+learned_sparse": (DENSE_ROUTE, sparse_route),
            "dense+bm25+learned_sparse": (DENSE_ROUTE, LEXICAL_ROUTE, sparse_route),
        }
        weighted = {
            name: _weighted_cross_validation(
                query_ids=scoped_query_ids,
                relevant=scoped_relevant,
                languages=scoped_languages,
                normalized=scoped_normalized,
                route_ids=route_ids,
            )
            for name, route_ids in configurations.items()
        }
        rrf = {
            name: _rrf_result(route_ids, scoped_routes, scoped_relevant, scoped_languages)
            for name, route_ids in configurations.items()
        }
        baseline = weighted["dense+bm25"]["fold_ndcg_at_10"]
        for result in weighted.values():
            result["fold_delta_vs_dense_bm25"] = [
                round(value - reference, 8)
                for value, reference in zip(result["fold_ndcg_at_10"], baseline, strict=True)
            ]
        report["studies"].append(
            {
                "learned_sparse_route": sparse_route,
                "scope": scope or "all",
                "queries": len(scoped_query_ids),
                "atomic": {
                    route_id: _atomic_result(route_id, scoped_routes, scoped_relevant, scoped_languages)
                    for route_id in (DENSE_ROUTE, LEXICAL_ROUTE, sparse_route)
                },
                "weighted": weighted,
                "rrf": rrf,
            }
        )
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Dense, BM25, and learned-sparse ablation",
        "",
        (
            "This study reuses frozen Top-50 route evidence. It performs no embedding, indexing, "
            "or retrieval calls. Weighted fusion uses per-query min-max normalization, a 0.05 "
            "simplex grid, and five-fold out-of-fold evaluation."
        ),
        "",
        f"Dense baseline: Voyage 4 Large, NDCG@10 `{report['dense']['metrics']['ndcg@10']:.4f}`.",
        "",
    ]
    for study in report["studies"]:
        sparse = study["learned_sparse_route"]
        languages = sorted(study["atomic"][DENSE_ROUTE]["language_metrics"])
        language_headers = " | ".join(f"Weighted {language.upper()}" for language in languages)
        lines.extend(
            [
                f"## {sparse} ({study['scope']}, {study['queries']} queries)",
                "",
                (
                    f"Atomic NDCG@10: Dense `{study['atomic'][DENSE_ROUTE]['metrics']['ndcg@10']:.4f}`, "
                    f"BM25 `{study['atomic'][LEXICAL_ROUTE]['metrics']['ndcg@10']:.4f}`, "
                    f"learned sparse `{study['atomic'][sparse]['metrics']['ndcg@10']:.4f}`."
                ),
                "",
                (
                    "| Configuration | RRF NDCG@10 | Weighted OOF NDCG@10 | "
                    f"{language_headers} | Full-data best weights (Dense/BM25/Sparse) |"
                ),
                "| --- | ---: | ---: | " + " | ".join("---:" for _ in languages) + " | --- |",
            ]
        )
        for name in ("dense+bm25", "dense+learned_sparse", "dense+bm25+learned_sparse"):
            weighted = study["weighted"][name]
            weights = weighted["full_best_weights"]
            if len(weights) == 2:
                display = "/".join(f"{value:.2f}" for value in weights)
            else:
                display = "/".join(f"{value:.2f}" for value in weights)
            language_values = " | ".join(
                f"{weighted['language_metrics'][language]['ndcg@10']:.4f}" for language in languages
            )
            lines.append(
                f"| {name} | {study['rrf'][name]['metrics']['ndcg@10']:.4f} | "
                f"{weighted['metrics']['ndcg@10']:.4f} | "
                f"{language_values} | `{display}` |"
            )
        three_way = study["weighted"]["dense+bm25+learned_sparse"]
        deltas = three_way["fold_delta_vs_dense_bm25"]
        lines.extend(
            [
                "",
                "Three-way weighted fold deltas versus Dense + BM25: "
                + ", ".join(f"`{value:+.4f}`" for value in deltas)
                + ".",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation rule",
            "",
            (
                "Learned sparse has incremental fusion value only when Dense + BM25 + learned sparse "
                "beats Dense + BM25 out of fold. A zero selected sparse weight is evidence that the "
                "stored learned-sparse route adds no measurable value under this fusion method."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    arguments = parser.parse_args()
    report = build_report(arguments.bundle)
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    arguments.markdown.write_text(_markdown(report))


if __name__ == "__main__":
    main()
