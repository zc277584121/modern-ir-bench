from __future__ import annotations

from benchmarks.modern_ir_release import _target_recall


def test_target_recall_only_scores_relevant_documents_in_the_selected_group() -> None:
    ranked_ids = ["other-relevant", "target-a", "irrelevant", "target-b"]
    target_ids = {"target-a", "target-b", "target-missed"}

    assert _target_recall(ranked_ids, target_ids, 1) == 0.0
    assert _target_recall(ranked_ids, target_ids, 2) == 1 / 3
    assert _target_recall(ranked_ids, target_ids, 4) == 2 / 3
