from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from benchmarks.ranking_evidence import load_run_artifact, merge_release, write_run_artifacts
from modern_ir_bench import MetricSet, RunProvenance
from modern_ir_bench.metrics import NDCG, Recall
from modern_ir_bench.results import RankingArtifact
from modern_ir_bench.solutions import CharacterNGramSolution, InMemoryBM25Solution
from modern_ir_bench.tasks import RankedRetrieval


def _dataset() -> DatasetDict:
    return DatasetDict(
        documents=Dataset.from_dict(
            {
                "document_id": ["d1", "d2", "d3"],
                "content": ["alpha beta", "gamma delta", "alpha gamma"],
            },
            features=RankedRetrieval.dataset_features["documents"],
        ),
        queries=Dataset.from_dict(
            {
                "query_id": ["q1", "q2"],
                "query": ["alpha beta", "gamma delta"],
            },
            features=RankedRetrieval.dataset_features["queries"],
        ),
        qrels=Dataset.from_dict(
            {
                "query_id": ["q1", "q2"],
                "document_id": ["d1", "d2"],
                "relevance": [1, 1],
            },
            features=RankedRetrieval.dataset_features["qrels"],
        ),
    )


def _task() -> RankedRetrieval:
    return RankedRetrieval(
        id="ranked-retrieval-test",
        title="Ranked Retrieval Test",
        description="Test task",
        version="1",
        datasets={"tiny": _dataset()},
        dataset_versions={"tiny": "1"},
        metrics=MetricSet(primary=NDCG(k=2), secondary=[Recall(k=2)]),
        top_k=2,
    )


def _provenance() -> RunProvenance:
    return RunProvenance(
        source_commit="0123456789abcdef",
        source_module="tests.benchmarks.test_ranking_evidence",
        source_path="tests/benchmarks/test_ranking_evidence.py",
        source_line=1,
        repository_url="https://github.com/example/modern-ir-bench",
        created_at="2026-10-12T00:00:00+00:00",
    )


def _write_run(tmp_path: Path, solution) -> Path:
    report = _task().run([solution], provenance=_provenance(), status="test")
    return write_run_artifacts(
        report,
        output_root=tmp_path / "runs",
        release="tiny",
        dataset_repository="example/data",
        dataset_revision="revision-1",
    )[0]


def test_solution_runs_merge_incrementally_into_one_ranking_release(tmp_path: Path) -> None:
    bm25_run = _write_run(tmp_path, InMemoryBM25Solution(id="bm25", title="BM25"))
    manifest, saved, observations = load_run_artifact(bm25_run)

    assert manifest["solution"] == "bm25"
    assert saved["solution"]["title"] == "BM25"
    assert len(observations) == 2

    output = tmp_path / "release"
    merge_release(
        task=_task(),
        dataset_id="tiny",
        dataset_repository="example/data",
        dataset_revision="revision-1",
        run_directories=[bm25_run],
        output=output,
    )
    manifest_path = output / "manifest.json"
    historical_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    historical_manifest["artifact_type"] = "historical-top-k-rankings"
    historical_manifest.pop("source_runs")
    manifest_path.write_text(json.dumps(historical_manifest), encoding="utf-8")

    ngram_run = _write_run(tmp_path, CharacterNGramSolution(id="ngram", title="N-gram"))
    merge_release(
        task=_task(),
        dataset_id="tiny",
        dataset_repository="example/data",
        dataset_revision="revision-1",
        run_directories=[ngram_run],
        output=output,
    )

    artifact = RankingArtifact.load(output)
    assert set(artifact.rankings) == {"bm25", "ngram"}
    assert set(artifact.manifest["source_runs"]) == {"bm25", "ngram"}
    assert artifact.manifest["source_runs"]["bm25"]["origin"] == "historical-top-k-rankings"
