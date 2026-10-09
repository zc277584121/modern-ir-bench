from __future__ import annotations

import json
from pathlib import Path

from modern_ir_bench.datasets import load_ranked_retrieval_release
from modern_ir_bench.retrieval.types import RetrievalResource
from modern_ir_bench.solutions import load_release_replay_solutions


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_release_loader_and_saved_ranking_solution(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {
                "doc_id": "d1",
                "title": "One",
                "text": "alpha",
                "dimensions": {"language": "en"},
            },
            {
                "doc_id": "d2",
                "title": "Two",
                "text": "beta",
                "dimensions": {"language": "zh"},
            },
        ],
    )
    _write_jsonl(
        tmp_path / "queries.jsonl",
        [{"query_id": "q1", "text": "alpha", "language": "en"}],
    )
    _write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"query_id": "q1", "doc_id": "d1", "relevance": 1}],
    )
    _write_jsonl(
        tmp_path / "baseline-observations.jsonl",
        [
            {
                "query_id": "q1",
                "solution_id": "saved",
                "ranked_doc_ids": ["d1", "d2"],
            }
        ],
    )
    (tmp_path / "baseline-results.json").write_text(
        json.dumps(
            {
                "solutions": {
                    "saved": {"title": "Saved", "route": "saved_route"},
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = load_ranked_retrieval_release(tmp_path)
    solution = load_release_replay_solutions(tmp_path)[0]
    resources = [RetrievalResource(id=row["document_id"], value=row["content"]) for row in dataset["documents"]]
    session = solution.prepare(resources)

    assert dataset["documents"][0]["content"] == "One\n\nalpha"
    assert [hit.id for hit in session.search_batch(["alpha"], top_k=1)[0]] == ["d1"]

    english = load_ranked_retrieval_release(tmp_path, language="en")
    assert english["documents"]["document_id"] == ["d1"]
    assert english["queries"]["query_id"] == ["q1"]
