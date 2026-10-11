from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Value

from modern_ir_bench import MetricSet
from modern_ir_bench.core.solution import Solution
from modern_ir_bench.metrics import NDCG, Recall
from modern_ir_bench.retrieval import PdfInput, RetrievalResource, SearchHit, TextInput
from modern_ir_bench.solutions import PaddleOcrExtractor, PdfOcrRetrievalSolution
from modern_ir_bench.tasks import PdfRetrieval


@dataclass(frozen=True)
class StubExtractor:
    id: str = "stub-pdf-text"

    def extract(self, value: PdfInput) -> str:
        assert value.language == "en"
        return "harbor permit renewal"


class StaticSession:
    metadata: Mapping[str, Any] = {"backend": "static"}

    def search_batch(self, queries: Sequence[TextInput], *, top_k: int) -> list[list[SearchHit]]:
        assert queries == [TextInput(text="permit renewal", language="en")]
        return [[SearchHit(id="doc-1", score=1.0)][:top_k]]

    def close(self) -> None:
        return None


class StubOcrPipeline:
    def __init__(self) -> None:
        self.calls = 0

    def predict_iter(self, input: str, **kwargs: Any) -> Iterable[Any]:
        assert Path(input).is_file()
        self.calls += 1
        yield {"res": {"rec_texts": ["first line", "low score"], "rec_scores": [0.99, 0.1]}}
        yield {"res": {"rec_texts": ["second page"], "rec_scores": [0.95]}}


def test_paddle_ocr_extracts_every_page_and_caches_by_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    pipeline = StubOcrPipeline()
    extractor = PaddleOcrExtractor(cache_dir=tmp_path / "cache", recognition_score_threshold=0.5)
    extractor._pipeline = pipeline
    value = PdfInput(path=str(pdf_path), language="en")

    first = extractor.extract(value)
    second = extractor.extract(value)

    assert first == "first line\n\nsecond page"
    assert second == first
    assert pipeline.calls == 1


@dataclass(frozen=True, kw_only=True)
class RecordingTextSolution(Solution):
    def prepare(self, resources: Iterable[RetrievalResource[TextInput]]) -> StaticSession:
        rows = list(resources)
        assert rows == [
            RetrievalResource(
                id="doc-1",
                value=TextInput(text="harbor permit renewal", language="en"),
            )
        ]
        return StaticSession()


def test_pdf_task_reuses_ranked_retrieval_with_pdf_resources(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc-1.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    dataset = DatasetDict(
        documents=Dataset.from_dict(
            {
                "document_id": ["doc-1"],
                "pdf": [str(pdf_path)],
                "language": ["en"],
            },
            features=Features(
                {
                    **PdfRetrieval.dataset_features["documents"],
                    "language": Value("string"),
                }
            ),
        ),
        queries=Dataset.from_dict(
            {
                "query_id": ["query-1"],
                "query": ["permit renewal"],
                "language": ["en"],
            },
            features=Features(
                {
                    **PdfRetrieval.dataset_features["queries"],
                    "language": Value("string"),
                }
            ),
        ),
        qrels=Dataset.from_dict(
            {"query_id": ["query-1"], "document_id": ["doc-1"], "relevance": [1]},
            features=PdfRetrieval.dataset_features["qrels"],
        ),
    )
    task = PdfRetrieval(
        id="pdf-retrieval",
        title="PDF Retrieval",
        description="test",
        version="1.0.0",
        datasets={"sample": dataset},
        dataset_versions={"sample": "test"},
        metrics=MetricSet(primary=NDCG(k=10), secondary=[Recall(k=1)]),
        top_k=1,
        document_value=None,
        query_value=lambda row: TextInput(text=row["query"], language=row["language"]),
        pool_field="language",
    )
    task.validate_dataset(dataset)
    solution = PdfOcrRetrievalSolution(
        id="pdf-text-static",
        title="PDF text static",
        text_solution=RecordingTextSolution(id="static", title="Static"),
        extractor=StubExtractor(),
    )

    observations = task.evaluate(dataset=dataset, solution=solution)

    assert observations[0]["query_id"] == "query-1"
    assert observations[0]["ranked_ids"] == ["doc-1"]
    assert observations[0]["expected_ids"] == ["doc-1"]
