"""Ranked retrieval where corpus resources are PDF documents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from datasets import Dataset, Features, Pdf, Value

from modern_ir_bench.retrieval import PdfInput
from modern_ir_bench.tasks.ranked_retrieval import RankedRetrieval


def pdf_input(row: Mapping[str, Any]) -> PdfInput:
    """Convert an undecoded Hugging Face Pdf value into the runtime input."""

    payload = row["pdf"]
    if not isinstance(payload, Mapping):
        raise TypeError("PdfRetrieval requires Pdf(decode=False)")
    return PdfInput(
        path=payload.get("path"),
        data=payload.get("bytes"),
        language=row.get("language"),
    )


class PdfRetrieval(RankedRetrieval):
    """Retrieve whole PDF documents and score their document identifiers."""

    dataset_features: ClassVar[dict[str, Features]] = {
        "documents": Features(
            {
                "document_id": Value("string"),
                "pdf": Pdf(decode=False),
            }
        ),
        "queries": RankedRetrieval.dataset_features["queries"],
        "qrels": RankedRetrieval.dataset_features["qrels"],
    }
    document_content_field: ClassVar[str | None] = None

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("document_value") is None:
            kwargs["document_value"] = pdf_input
        super().__init__(**kwargs)

    def validate_document_rows(self, documents: Dataset) -> None:
        for row in documents:
            payload = row["pdf"]
            if not isinstance(payload, Mapping) or (not payload.get("path") and payload.get("bytes") is None):
                raise ValueError(f"PDF payload is empty for {row['document_id']}")
