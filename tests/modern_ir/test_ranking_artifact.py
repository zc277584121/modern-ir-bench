from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from modern_ir_bench.results import RankingArtifact


def test_ranking_artifact_is_data_not_a_solution(tmp_path: Path) -> None:
    ranking_path = tmp_path / "rankings.jsonl.gz"
    with gzip.open(ranking_path, "wt", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "query_id": "q1",
                    "solution_id": "example",
                    "ranked_doc_ids": ["d2", "d1"],
                }
            )
            + "\n"
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "historical-top-k-rankings",
                "solutions": ["example"],
                "queries": 1,
                "top_k": 2,
                "ranking_rows": 1,
                "rankings_sha256": hashlib.sha256(ranking_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    artifact = RankingArtifact.load(tmp_path)

    assert artifact.rankings == {"example": {"q1": ("d2", "d1")}}
    assert not hasattr(artifact, "prepare")
