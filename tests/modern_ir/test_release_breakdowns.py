from __future__ import annotations

from benchmarks.reporting import Dimension, build_retrieval_breakdowns


def test_relevant_document_slice_only_scores_selected_targets() -> None:
    breakdowns = build_retrieval_breakdowns(
        task_id="retrieval",
        dataset_id="sample",
        dataset_version="1",
        rankings={"solution": {"q": ("other-relevant", "target-a", "irrelevant", "target-b")}},
        relevant_by_query={"q": {"other-relevant", "target-a", "target-b", "target-missed"}},
        query_labels={"q": {"language": "en"}},
        document_labels={
            "other-relevant": {"group": "other"},
            "target-a": {"group": "target"},
            "target-b": {"group": "target"},
            "target-missed": {"group": "target"},
        },
        document_dimensions=(Dimension("group", "Group", (("target", "Target"),)),),
    )

    results = breakdowns[0]["values"][0]["results"]
    scores = {row["metric_id"]: row["value"] for row in results}
    assert scores == {
        "target_recall@1": 0.0,
        "target_recall@5": 0.66666667,
        "target_recall@10": 0.66666667,
    }
