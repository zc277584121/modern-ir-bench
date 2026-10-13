"""Build a reproducible diagnostic report for the frozen text benchmark release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.ranked_retrieval.definition import RELEASE_ID, _document_value, _query_value, build_task
from benchmarks.ranked_retrieval.solutions.hybrid import CANDIDATE_CACHE_NAMESPACE
from benchmarks.reporting import retrieval_metrics
from modern_ir_bench.results import RankingArtifact
from modern_ir_bench.retrieval import RetrievalResource
from modern_ir_bench.retrieval.candidate_cache import SqliteCandidateCache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results/modern-ir-v2.2"
DEFAULT_CACHE = PROJECT_ROOT / ".runtime/retrieval-candidates.sqlite3"

CANDIDATE_PATHS = {
    "qwen3-embedding-8b": "internal-qwen3-embedding-8b-chunk",
    "bm25-qwen3-embedding-8b-weighted-25-75": "internal-bm25-qwen3-embedding-8b-weighted-25-75",
}
COMPARISONS = (
    (
        "qwen3-embedding-8b-openrouter-chunk",
        "qwen3-embedding-8b-rerank-3",
        "Qwen3 Embedding 8B → Voyage Rerank 3",
    ),
    (
        "bm25-qwen3-embedding-8b-weighted-25-75",
        "bm25-qwen3-embedding-8b-weighted-rerank-3",
        "Weighted candidate path → Voyage Rerank 3",
    ),
    (
        "qwen3-embedding-8b-openrouter-chunk",
        "qwen3-embedding-8b-bge-reranker-v2-m3",
        "Qwen3 Embedding 8B → BGE Reranker v2 M3",
    ),
    (
        "bm25-qwen3-embedding-8b-weighted-25-75",
        "bm25-qwen3-embedding-8b-weighted-bge-reranker-v2-m3",
        "Weighted candidate path → BGE Reranker v2 M3",
    ),
)
SLICE_FIELDS = ("language", "query_intent", "expression", "constraint_level")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relevant_documents(qrels: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        if int(row["relevance"]) > 0:
            relevant[str(row["query_id"])].add(str(row["document_id"]))
    return dict(relevant)


def _load_candidates(dataset: Any, cache_path: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    cache = SqliteCandidateCache(cache_path, namespace=CANDIDATE_CACHE_NAMESPACE)
    candidates = {path_id: {} for path_id in CANDIDATE_PATHS}
    for language in sorted(set(dataset["queries"]["language"])):
        documents = [
            RetrievalResource(id=str(row["document_id"]), value=_document_value(row))
            for row in dataset["documents"]
            if row["language"] == language
        ]
        queries = [row for row in dataset["queries"] if row["language"] == language]
        corpus_key = cache.corpus_key(documents)
        query_values = [_query_value(row) for row in queries]
        for path_id, component_id in CANDIDATE_PATHS.items():
            cached = cache.read(
                corpus_key=corpus_key,
                component_id=component_id,
                queries=query_values,
                depth=50,
            )
            if len(cached) != len(queries):
                raise RuntimeError(
                    f"Candidate cache is incomplete for {path_id}/{language}: {len(cached)}/{len(queries)}"
                )
            candidates[path_id].update(
                {
                    str(row["query_id"]): tuple(hit.id for hit in cached[index])
                    for index, row in enumerate(queries)
                }
            )
    return candidates


def _write_candidate_evidence(
    path: Path,
    candidates: Mapping[str, Mapping[str, Sequence[str]]],
) -> None:
    temporary = path.with_suffix(".tmp.gz")
    with (
        temporary.open("wb") as raw_stream,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as stream,
    ):
        for path_id in sorted(candidates):
            for query_id in sorted(candidates[path_id]):
                stream.write(
                    json.dumps(
                        {
                            "candidate_path": path_id,
                            "query_id": query_id,
                            "ranked_doc_ids": list(candidates[path_id][query_id]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    temporary.replace(path)


def _candidate_summary(
    candidates: Mapping[str, Mapping[str, Sequence[str]]],
    relevant: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    summaries = []
    for path_id, rankings in candidates.items():
        recalls = [len(relevant[query_id].intersection(ranked)) / len(relevant[query_id]) for query_id, ranked in rankings.items()]
        summaries.append(
            {
                "candidate_path": path_id,
                "depth": 50,
                "macro_recall": round(statistics.mean(recalls), 8),
                "queries_with_any_relevant": round(
                    statistics.mean(bool(relevant[query_id].intersection(ranked)) for query_id, ranked in rankings.items()),
                    8,
                ),
                "queries_with_all_relevant": round(
                    statistics.mean(relevant[query_id].issubset(ranked) for query_id, ranked in rankings.items()),
                    8,
                ),
            }
        )
    return summaries


def _comparison_summary(
    *,
    before_id: str,
    after_id: str,
    label: str,
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    relevant: Mapping[str, set[str]],
    query_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    before = {
        query_id: retrieval_metrics(relevant[query_id], ranked)
        for query_id, ranked in rankings[before_id].items()
    }
    after = {
        query_id: retrieval_metrics(relevant[query_id], ranked)
        for query_id, ranked in rankings[after_id].items()
    }
    deltas = {query_id: after[query_id]["ndcg@10"] - before[query_id]["ndcg@10"] for query_id in relevant}
    outcomes = Counter(
        "improved" if delta > 1e-12 else "regressed" if delta < -1e-12 else "unchanged"
        for delta in deltas.values()
    )
    slices = {}
    for field in SLICE_FIELDS:
        slices[field] = [
            {
                "value": value,
                "queries": len(selected),
                "ndcg_delta": round(statistics.mean(deltas[query_id] for query_id in selected), 8),
            }
            for value in sorted({str(row[field]) for row in query_rows.values()})
            if (selected := [query_id for query_id, row in query_rows.items() if str(row[field]) == value])
        ]

    def examples(reverse: bool) -> list[dict[str, Any]]:
        selected = sorted(deltas, key=lambda query_id: (deltas[query_id], query_id), reverse=reverse)[:5]
        return [
            {
                "query_id": query_id,
                "query": str(query_rows[query_id]["query"]),
                "language": str(query_rows[query_id]["language"]),
                "ndcg_before": round(before[query_id]["ndcg@10"], 8),
                "ndcg_after": round(after[query_id]["ndcg@10"], 8),
                "ndcg_delta": round(deltas[query_id], 8),
            }
            for query_id in selected
        ]

    return {
        "label": label,
        "before": before_id,
        "after": after_id,
        "ndcg_delta": round(statistics.mean(deltas.values()), 8),
        "outcomes": dict(sorted(outcomes.items())),
        "slices": slices,
        "largest_improvements": examples(True),
        "largest_regressions": examples(False),
    }


def _difficulty_summary(
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    relevant: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    solution_ids = (
        "qwen3-embedding-8b-rerank-3",
        "bm25-qwen3-embedding-8b-weighted-rerank-3",
    )
    return [
        {
            "solution": solution_id,
            "first_rank_relevant": round(
                statistics.mean(ranked[0] in relevant[query_id] for query_id, ranked in rankings[solution_id].items()),
                8,
            ),
            "all_relevant_in_top_10": round(
                statistics.mean(
                    relevant[query_id].issubset(ranked) for query_id, ranked in rankings[solution_id].items()
                ),
                8,
            ),
        }
        for solution_id in solution_ids
    ]


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Modern IR Ranked Retrieval v2.2 diagnostics",
        "",
        "This report is derived from the frozen Top-10 release and the exported Top-50 candidate evidence.",
        "It diagnoses candidate coverage, reranker behavior, and benchmark difficulty without changing Qrels.",
        "",
        "## Candidate ceiling",
        "",
        "| Candidate path | Recall@50 | Any relevant | All relevant |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["candidate_paths"]:
        lines.append(
            f"| `{row['candidate_path']}` | {row['macro_recall']:.4f} | "
            f"{row['queries_with_any_relevant']:.1%} | {row['queries_with_all_relevant']:.1%} |"
        )
    lines.extend(["", "## Reranker effect", ""])
    for comparison in report["comparisons"]:
        outcomes = comparison["outcomes"]
        lines.append(
            f"- **{comparison['label']}**: ΔNDCG@10 `{comparison['ndcg_delta']:+.4f}`; "
            f"{outcomes.get('improved', 0)} improved, {outcomes.get('regressed', 0)} regressed, "
            f"{outcomes.get('unchanged', 0)} unchanged."
        )
    lines.extend(
        [
            "",
            "## Difficulty signal",
            "",
            "| Solution | Relevant at rank 1 | All relevant in Top 10 |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in report["difficulty"]:
        lines.append(
            f"| `{row['solution']}` | {row['first_rank_relevant']:.1%} | {row['all_relevant_in_top_10']:.1%} |"
        )
    lines.extend(
        [
            "",
            (
                "The first positive is easy for the strongest reranker, but complete positive-set recall is not. "
                "NDCG@10 and Recall@10 should therefore be read together; MRR alone overstates saturation."
            ),
            "",
            "Detailed slices and concrete best/worst queries are stored in `analysis.json`.",
            "Top-50 evidence is stored in `candidate-rankings.jsonl.gz`; its checksum is recorded in `analysis.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(*, result_root: Path, candidate_cache: Path) -> dict[str, Any]:
    artifact = RankingArtifact.load(result_root)
    task = build_task()
    dataset = task.datasets[RELEASE_ID]
    relevant = _relevant_documents(dataset["qrels"])
    query_rows = {str(row["query_id"]): row for row in dataset["queries"]}
    candidates = _load_candidates(dataset, candidate_cache)
    evidence_path = result_root / "candidate-rankings.jsonl.gz"
    _write_candidate_evidence(evidence_path, candidates)
    report = {
        "schema_version": 1,
        "release": RELEASE_ID,
        "ranking_evidence_sha256": artifact.manifest["rankings_sha256"],
        "candidate_evidence_sha256": _sha256(evidence_path),
        "queries": len(relevant),
        "positive_qrels": sum(map(len, relevant.values())),
        "mean_positive_qrels_per_query": round(statistics.mean(map(len, relevant.values())), 8),
        "candidate_paths": _candidate_summary(candidates, relevant),
        "comparisons": [
            _comparison_summary(
                before_id=before_id,
                after_id=after_id,
                label=label,
                rankings=artifact.rankings,
                relevant=relevant,
                query_rows=query_rows,
            )
            for before_id, after_id, label in COMPARISONS
        ],
        "difficulty": _difficulty_summary(artifact.rankings, relevant),
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
    parser.add_argument("--candidate-cache", type=Path, default=DEFAULT_CACHE)
    arguments = parser.parse_args()
    report = build_report(
        result_root=arguments.result_root.resolve(),
        candidate_cache=arguments.candidate_cache.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
