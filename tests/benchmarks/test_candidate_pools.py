from __future__ import annotations

import pytest

from benchmarks.candidate_pools import CandidateGroup, analyze_benchmark


def test_analyze_benchmark_measures_route_and_union_coverage() -> None:
    benchmark = {
        "task_id": "task",
        "dataset_id": "dataset",
        "dataset_version": "revision",
        "max_depth": 2,
        "queries": [
            {"id": "q1", "relevant": [0, 1]},
            {"id": "q2", "relevant": [2]},
        ],
        "routes": [
            {"id": "dense", "label": "Dense"},
            {"id": "sparse", "label": "Sparse"},
        ],
    }
    report = analyze_benchmark(
        benchmark,
        {
            "dense": [(0, 3), (2, 4)],
            "sparse": [(1, 3), (4, 5)],
        },
        (CandidateGroup("union", "Dense + sparse", ("dense", "sparse")),),
    )

    dense, sparse = report["routes"]
    assert dense["relevant_coverage"] == 0.75
    assert dense["globally_unique_relevant"] == 2
    assert sparse["relevant_coverage"] == 0.25
    assert sparse["globally_unique_relevant"] == 1
    assert report["groups"][0]["mean_candidates"] == 3
    assert report["groups"][0]["relevant_coverage"] == 1.0
    assert report["groups"][0]["oracle_ndcg_at_10"] == 1.0
    assert report["pairs"][0] == {
        "left": "dense",
        "right": "sparse",
        "mean_jaccard": pytest.approx(1 / 3),
        "left_unique_relevant": 2,
        "right_unique_relevant": 1,
    }


def test_analyze_benchmark_rejects_unknown_group_route() -> None:
    benchmark = {
        "task_id": "task",
        "dataset_id": "dataset",
        "dataset_version": "revision",
        "max_depth": 2,
        "queries": [{"id": "q", "relevant": [0]}],
        "routes": [{"id": "dense", "label": "Dense"}],
    }

    with pytest.raises(ValueError, match="unknown routes"):
        analyze_benchmark(
            benchmark,
            {"dense": [(0, 1)]},
            (CandidateGroup("bad", "Bad", ("missing",)),),
        )
