from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from benchmarks.languages import select_solutions
from benchmarks.pdf_retrieval import definition as pdf_definition
from benchmarks.pdf_retrieval import run as pdf_run
from benchmarks.pdf_retrieval.solutions import SOLUTION_SPECS as PDF_SPECS
from benchmarks.publish import _release_report
from benchmarks.ranked_retrieval import definition as text_definition
from benchmarks.ranked_retrieval import run as text_run
from benchmarks.ranked_retrieval.solutions import SOLUTION_SPECS as TEXT_SPECS
from benchmarks.ranking_evidence import merge_release, validate_language_slice, write_run_artifacts
from benchmarks.reporting import Dimension
from modern_ir_bench import MetricSet, RunProvenance
from modern_ir_bench.exporters import build_space_payload
from modern_ir_bench.metrics import NDCG, Recall
from modern_ir_bench.results import RankingArtifact
from modern_ir_bench.solutions import InMemoryBM25Solution
from modern_ir_bench.tasks import RankedRetrieval

ENGLISH_TEXT = {
    "splade-v3-distilbert-sparse-chunk",
    "splade-v3-full-sparse-chunk",
    "opensearch-doc-v2-distill-sparse-chunk",
    "granite-30m-sparse-chunk",
}
ENGLISH_PDF = {
    "paddleocr-v5-mobile-colbert-v2-exact-maxsim-chunks",
    "colsmol-256m-page-exact-maxsim",
    "colsmol-500m-page-exact-maxsim",
    "colpali-v1-3-page-exact-maxsim",
}
COVERAGE = {"bilingual": ("en", "zh"), "english": ("en",)}
REVISION = "dataset-revision"


def release_id(language=None):
    return "tiny" if language is None else f"tiny-language-{language}"


def task_factory(*, language=None):
    documents = Dataset.from_list(
        [
            {"document_id": "e1", "content": "alpha beta", "language": "en", "group": "a"},
            {"document_id": "e2", "content": "gamma delta", "language": "en", "group": "b"},
            {"document_id": "z1", "content": "alpha beta", "language": "zh", "group": "a"},
            {"document_id": "z2", "content": "gamma delta", "language": "zh", "group": "b"},
        ]
    )
    queries = Dataset.from_list(
        [
            {"query_id": "qe", "query": "alpha beta", "language": "en"},
            {"query_id": "qz", "query": "gamma delta", "language": "zh"},
        ]
    )
    qrels = Dataset.from_list(
        [
            {"query_id": "qe", "document_id": "e1", "relevance": 1},
            {"query_id": "qz", "document_id": "z2", "relevance": 1},
        ]
    )
    qrels = qrels.cast(RankedRetrieval.dataset_features["qrels"])
    if language is not None:
        documents = documents.filter(lambda row: row["language"] == language)
        queries = queries.filter(lambda row: row["language"] == language)
        qrels = qrels.filter(lambda row: row["query_id"] in set(queries["query_id"]))
    return RankedRetrieval(
        id="test",
        title="Test",
        description="Test",
        version="1",
        datasets={release_id(language): DatasetDict(documents=documents, queries=queries, qrels=qrels)},
        dataset_versions={release_id(language): REVISION[:12]},
        metrics=MetricSet(primary=NDCG(k=10), secondary=[Recall(k=10)]),
        top_k=2,
        pool_field="language",
    )


def write_run(tmp_path, solution_id="english", language=None):
    provenance = RunProvenance(
        source_commit="a" * 40,
        source_module="test",
        source_path="test.py",
        source_line=1,
        repository_url="https://github.com/example/benchmark",
        created_at="2026-10-01T00:00:00+00:00",
    )
    report = task_factory(language=language).run(
        [InMemoryBM25Solution(id=solution_id, title=solution_id)],
        provenance=provenance,
    )
    return write_run_artifacts(
        report,
        output_root=tmp_path / release_id(language),
        release=release_id(language),
        dataset_repository="example/data",
        dataset_revision=REVISION,
    )[0]


def merge(output, *, runs=(), derive=(), language=None, coverage=COVERAGE, **kwargs):
    merge_release(
        task=task_factory(language=language),
        dataset_id=release_id(language),
        dataset_repository="example/data",
        dataset_revision=REVISION,
        run_directories=runs,
        output=output,
        solution_languages=coverage,
        derive_from=derive,
        source_task=task_factory() if derive else None,
        source_dataset_id=release_id() if derive else None,
        **kwargs,
    )


@pytest.mark.parametrize("specs,english", [(TEXT_SPECS, ENGLISH_TEXT), (PDF_SPECS, ENGLISH_PDF)])
def test_catalog_and_default_selection_use_explicit_coverage(specs, english):
    assert {spec.id for spec in specs if spec.languages == ("en",)} == english
    assert all(spec.languages in (("en",), ("en", "zh")) for spec in specs)
    assert {spec.id for spec in specs if "english-only" in spec.tags} == english
    assert set(select_solutions(specs, None, None)) == {spec.id for spec in specs} - english
    assert set(select_solutions(specs, None, "en")) == {spec.id for spec in specs}
    assert set(select_solutions(specs, None, "zh")) == {spec.id for spec in specs} - english


@pytest.mark.parametrize(
    "runner,solution",
    [
        (text_run, "granite-30m-sparse-chunk"),
        (pdf_run, "colsmol-500m-page-exact-maxsim"),
    ],
)
@pytest.mark.parametrize("language", [None, "zh"])
def test_runner_rejects_unsupported_scope_before_loading_dataset_or_models(monkeypatch, runner, solution, language):
    monkeypatch.setattr(runner, "_arguments", lambda: argparse.Namespace(solution=[solution], language=language))

    def unexpected(**kwargs):
        pytest.fail("Dataset/model construction must not occur")

    monkeypatch.setattr(runner, "build_task", unexpected)
    monkeypatch.setattr(runner, "build_solutions", unexpected)
    with pytest.raises(ValueError, match="requested scope requires"):
        runner.main()


def test_release_rejects_historical_english_run_before_writing(tmp_path):
    run = write_run(tmp_path)
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="requested scope requires"):
        merge(output, runs=[run])
    assert not output.exists()


@pytest.mark.parametrize("kind", ["run", "ranking"])
def test_slice_derivation_preserves_rankings_and_source_checksums(tmp_path, kind):
    run = write_run(tmp_path)
    source = run
    if kind == "ranking":
        source = tmp_path / "historical"
        merge(source, runs=[run], coverage={"english": ("en", "zh")})
    before = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    target = tmp_path / "slice"
    merge(target, derive=[source], language="en", derive_solutions=["english"])
    artifact = RankingArtifact.load(target)
    assert artifact.rankings == {"english": {"qe": ("e1", "e2")}}
    assert artifact.manifest["queries"] == 1
    assert artifact.manifest["languages"] == ["en"]
    provenance = artifact.manifest["source_runs"]["english"]
    assert provenance["source_commit"] == "a" * 40
    origin = provenance["derivation"]["source_artifact"]
    assert origin["manifest_sha256"] == hashlib.sha256(before["manifest.json"]).hexdigest()
    checksum_file = "observations.parquet" if kind == "run" else "rankings.jsonl.gz"
    checksum_key = "observations_sha256" if kind == "run" else "rankings_sha256"
    assert origin[checksum_key] == hashlib.sha256(before[checksum_file]).hexdigest()
    assert before == {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}


def test_slice_derivation_rejects_corrupt_or_wrong_dataset_source(tmp_path):
    run = write_run(tmp_path)
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["dataset"]["revision"] = "other"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different Dataset"):
        merge(tmp_path / "slice", derive=[run], language="en")
    manifest["dataset"]["revision"] = REVISION
    manifest["observations_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum mismatch"):
        merge(tmp_path / "slice", derive=[run], language="en")
    assert not (tmp_path / "slice").exists()


def test_recomputed_full_metrics_are_checked_before_derivation(tmp_path):
    run = write_run(tmp_path)
    results_path = run / "results.json"
    results = json.loads(results_path.read_text())
    results["results"][0]["value"] = 0.123
    results_path.write_text(json.dumps(results))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["results_sha256"] = hashlib.sha256(results_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="aggregate metrics"):
        merge(tmp_path / "slice", derive=[run], language="en")


def publish(root):
    return _release_report(
        task_factory=task_factory,
        release_id=release_id,
        dataset_repository="example/data",
        dataset_revision=REVISION,
        root=root,
        result_path="results/test",
        all_metadata={
            sid: {"title": sid, "description": sid, "languages": list(langs)} for sid, langs in COVERAGE.items()
        },
        repository_url="https://github.com/example/benchmark",
        commit="b" * 40,
        query_dimensions=(Dimension("language", "Language", (("en", "English"), ("zh", "Chinese"))),),
        document_dimensions=(Dimension("group", "Group"),),
    )


def test_publication_only_adds_english_metrics_to_language_breakdown(tmp_path):
    root = tmp_path / "release"
    bilingual = write_run(tmp_path, "bilingual")
    english = write_run(tmp_path, "english", "en")
    merge(root, runs=[bilingual])
    merge(root / "slices/en", runs=[english], language="en")
    report, breakdowns, coverage = publish(root)
    assert set(report.solutions) == {"bilingual", "english"}
    assert {row["solution_id"] for row in report.records} == {"bilingual"}
    language = next(item for item in breakdowns if item["id"] == "language")
    en, zh = language["values"]
    assert {row["solution_id"] for row in en["results"]} == {"bilingual", "english"}
    assert {row["solution_id"] for row in zh["results"]} == {"bilingual"}
    en_scores = {row["metric_id"]: row["value"] for row in en["results"] if row["solution_id"] == "english"}
    bi_scores = {row["metric_id"]: row["value"] for row in en["results"] if row["solution_id"] == "bilingual"}
    assert en_scores == bi_scores
    assert zh["excluded_solutions"][0]["reason"] == "unsupported-language"
    assert coverage["excluded_solutions"][0]["solution_id"] == "english"
    assert next(item for item in coverage["solutions"] if item["solution_id"] == "english")["evidence"][0][
        "languages"
    ] == ["en"]
    for breakdown in breakdowns:
        if breakdown["id"] != "language":
            assert all(row["solution_id"] != "english" for value in breakdown["values"] for row in value["results"])


def test_publisher_rejects_english_solution_in_complete_release(tmp_path):
    root = tmp_path / "release"
    merge(root, runs=[write_run(tmp_path)], coverage={"english": ("en", "zh")})
    with pytest.raises(ValueError, match="requested scope requires"):
        publish(root)


def test_matching_bilingual_slice_does_not_duplicate_language_scores(tmp_path):
    root = tmp_path / "release"
    run = write_run(tmp_path, "bilingual")
    merge(root, runs=[run])
    merge(root / "slices/en", derive=[run], language="en")
    _, breakdowns, _ = publish(root)
    en = next(item for item in breakdowns if item["id"] == "language")["values"][0]
    assert len(en["results"]) == len({row["metric_id"] for row in en["results"]})


def rewrite_rankings(root, mutate):
    path = root / "rankings.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    mutate(rows)
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rankings_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("query_id", "unexpected-query", "every query exactly once"),
        ("ranked_doc_ids", ["z1", "z2"], "crosses independent pools"),
        ("ranked_doc_ids", ["unknown", "e2"], "unknown documents"),
        ("ranked_doc_ids", ["e1"], "wrong depth"),
    ],
)
def test_derivation_rejects_semantically_invalid_ranking_evidence(tmp_path, field, value, message):
    source = tmp_path / "source"
    merge(source, runs=[write_run(tmp_path)], coverage={"english": ("en", "zh")})
    rewrite_rankings(source, lambda rows: rows[0].__setitem__(field, value))
    with pytest.raises(ValueError, match=message):
        merge(tmp_path / "slice", derive=[source], language="en")
    assert not (tmp_path / "slice").exists()


def test_publisher_rejects_conflicting_full_and_slice_rankings(tmp_path):
    root = tmp_path / "release"
    run = write_run(tmp_path, "bilingual")
    merge(root, runs=[run])
    merge(root / "slices/en", derive=[run], language="en")
    rewrite_rankings(root / "slices/en", lambda rows: rows[0]["ranked_doc_ids"].reverse())
    with pytest.raises(ValueError, match="Conflicting full and language evidence"):
        publish(root)


def test_missing_derivation_selection_leaves_existing_release_unchanged(tmp_path):
    root = tmp_path / "release"
    run = write_run(tmp_path, "bilingual")
    merge(root, derive=[run], language="en")
    before = (root / "manifest.json").read_bytes(), (root / "rankings.jsonl.gz").read_bytes()
    with pytest.raises(ValueError, match="missing from the sources"):
        merge(root, derive=[run], language="en", derive_solutions=["english"])
    assert before == ((root / "manifest.json").read_bytes(), (root / "rankings.jsonl.gz").read_bytes())


@pytest.mark.parametrize(
    "definition,loader",
    [
        (text_definition, "load_ranked_retrieval_hub"),
        (pdf_definition, "load_pdf_retrieval_hub"),
    ],
)
def test_definitions_pin_dataset_revision_and_forward_language(monkeypatch, definition, loader):
    calls = []

    def load(repository, *, revision, language):
        calls.append((repository, revision, language))
        return "dataset"

    monkeypatch.setattr(definition, loader, load)
    task = definition.build_task(language="en")
    assert task.datasets == {definition.RELEASE_ID + "-language-en": "dataset"}
    assert calls == [(definition.DATASET_ID, definition.DATASET_REVISION, "en")]
    with pytest.raises(ValueError, match="language coverage"):
        definition.build_task(language="fr")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "split,field,value",
    [("queries", "query", "changed"), ("documents", "content", "changed"), ("qrels", "relevance", 2)],
)
def test_language_slice_requires_identical_evaluation_content(split, field, value):
    full = task_factory().datasets[release_id()]
    sliced = task_factory(language="en").datasets[release_id("en")]
    sliced[split] = sliced[split].map(lambda row: {field: value})
    with pytest.raises(ValueError, match=f"Language slice {split} differ"):
        validate_language_slice(full, sliced, "en")


def test_slice_derivation_requires_independent_language_pools(tmp_path):
    source = task_factory()
    source.pool_field = None
    with pytest.raises(ValueError, match="independent language retrieval pools"):
        merge_release(
            task=task_factory(language="en"),
            dataset_id=release_id("en"),
            dataset_repository="example/data",
            dataset_revision=REVISION,
            run_directories=[],
            output=tmp_path / "slice",
            solution_languages=COVERAGE,
            derive_from=[tmp_path / "unused"],
            source_task=source,
            source_dataset_id=release_id(),
        )
    assert not (tmp_path / "slice").exists()


def test_published_payload_matches_javascript_scope_contract(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node.js is required for the Space contract test")
    root = tmp_path / "release"
    merge(root, runs=[write_run(tmp_path, "bilingual")])
    merge(root / "slices/en", runs=[write_run(tmp_path, "english", "en")], language="en")
    report, breakdowns, coverage = publish(root)
    payload = build_space_payload(report, release="test", breakdowns=breakdowns, coverage=[coverage])
    script = """
        import assert from 'node:assert/strict';
        import { readFileSync } from 'node:fs';
        import { validatePayload, evaluateScope } from './space/scope.js';
        const payload = validatePayload(JSON.parse(readFileSync(0, 'utf8')));
        const context = payload.coverage[0];
        const ids = (options) => evaluateScope(payload, context, options).entries.map(({ solution }) => solution.id).sort();
        assert.deepEqual(ids(), ['bilingual']);
        assert.deepEqual(ids({breakdownId: 'language', valueId: 'en'}), ['bilingual', 'english']);
        assert.deepEqual(ids({breakdownId: 'language', valueId: 'zh'}), ['bilingual']);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "directory,specs,english,full_count",
    [
        ("modern-ir-v2.2", TEXT_SPECS, ENGLISH_TEXT - {"splade-v3-full-sparse-chunk"}, 43),
        ("modern-ir-pdf-v1", PDF_SPECS, ENGLISH_PDF, 13),
    ],
)
def test_migrated_releases_have_separate_complete_language_evidence(directory, specs, english, full_count):
    root = Path(__file__).resolve().parents[2] / "results" / directory
    full = RankingArtifact.load(root)
    sliced = RankingArtifact.load(root / "slices/en")
    supported = {spec.id: spec.languages for spec in specs}
    assert len(full.rankings) == full_count
    assert not english.intersection(full.rankings)
    assert all(set(supported[sid]) == {"en", "zh"} for sid in full.rankings)
    assert set(sliced.rankings) == english
    assert sliced.manifest["queries"] == 500
    assert sliced.manifest["languages"] == ["en"]
    full_queries = set(next(iter(full.rankings.values())))
    slice_queries = set(next(iter(sliced.rankings.values())))
    assert len(full_queries) == 1000 and slice_queries < full_queries
    assert all(set(rankings) == slice_queries for rankings in sliced.rankings.values())
    for source in sliced.manifest["source_runs"].values():
        original = source.get("derivation", source)["source_artifact"]
        assert len(original["manifest_sha256"]) == 64
        assert original["dataset"] == sliced.manifest["dataset"]
        assert original["top_k"] == 10


@pytest.mark.parametrize("split,indices", [("queries", [1]), ("documents", [0, 2]), ("qrels", [1])])
def test_language_slice_validates_selected_rows_not_underlying_arrow_storage(split, indices):
    full = task_factory().datasets[release_id()]
    sliced = task_factory(language="en").datasets[release_id("en")]
    sliced[split] = full[split].select(indices)
    with pytest.raises(ValueError, match=f"Language slice {split} differ"):
        validate_language_slice(full, sliced, "en")


def test_language_slice_accepts_different_storage_order():
    full = task_factory().datasets[release_id()]
    sliced = task_factory(language="en").datasets[release_id("en")]
    sliced["documents"] = Dataset.from_list(list(reversed(sliced["documents"].to_list())))
    validate_language_slice(full, sliced, "en")


@pytest.mark.parametrize("spec_type", [type(TEXT_SPECS[0]), type(PDF_SPECS[0])])
def test_solution_spec_requires_explicit_languages(spec_type):
    import inspect

    assert inspect.signature(spec_type).parameters["languages"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="languages"):
        spec_type(
            id="unclassified-model",
            title="Unclassified",
            description="Missing language declaration",
            tags=(),
            factory=lambda runtime, spec: None,
            readme_path="docs/solutions/example.md",
        )


@pytest.mark.parametrize("catalog", ["ranked_retrieval", "pdf_retrieval"])
def test_every_catalog_entry_declares_literal_language_coverage(catalog):
    import ast

    root = Path(__file__).resolve().parents[2] / "benchmarks" / catalog / "solutions"
    entries = 0
    for path in root.glob("*.py"):
        for call in ast.walk(ast.parse(path.read_text())):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "SolutionSpec":
                continue
            entries += 1
            declarations = [keyword.value for keyword in call.keywords if keyword.arg == "languages"]
            assert len(declarations) == 1, f"{path}:{call.lineno} needs explicit languages"
            assert ast.literal_eval(declarations[0]) in (("en", "zh"), ("en",))
    assert entries == len(TEXT_SPECS if catalog == "ranked_retrieval" else PDF_SPECS)
