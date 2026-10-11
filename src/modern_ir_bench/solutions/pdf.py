"""Composable PDF-to-text retrieval Solution."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.retrieval import PdfInput, RetrievalResource, SearchHit, TextInput


class PdfTextExtractor(Protocol):
    """Convert one PDF into searchable text."""

    @property
    def id(self) -> str: ...

    def extract(self, value: PdfInput) -> str: ...


@dataclass(frozen=True)
class PopplerTextExtractor:
    """Read only an existing PDF text layer; this intentionally does not run OCR."""

    executable: str = "pdftotext"
    timeout_seconds: int = 60

    @property
    def id(self) -> str:
        return "poppler-pdftotext-layout-v1"

    def extract(self, value: PdfInput) -> str:
        if value.path and Path(value.path).is_file():
            return self._extract_path(Path(value.path))
        if value.data is None:
            raise FileNotFoundError(f"PDF path is not accessible: {value.path}")
        with tempfile.NamedTemporaryFile(suffix=".pdf") as stream:
            stream.write(value.data)
            stream.flush()
            return self._extract_path(Path(stream.name))

    def _extract_path(self, path: Path) -> str:
        result = subprocess.run(
            [self.executable, "-layout", "-enc", "UTF-8", str(path), "-"],
            check=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        return result.stdout.decode("utf-8", errors="replace").replace("\f", "\n").strip()


@dataclass(frozen=True, kw_only=True)
class PdfTextExtractionSolution(Solution):
    """Extract PDF text, then delegate indexing and search to a text Solution."""

    text_solution: Solution
    extractor: PdfTextExtractor = field(default_factory=PopplerTextExtractor)
    empty_document_text: str = "[no extractable PDF text]"

    def prepare(
        self,
        resources: Iterable[RetrievalResource[PdfInput]],
    ) -> PdfTextExtractionSession:
        extracted: list[RetrievalResource[TextInput]] = []
        empty_documents = 0
        for resource in resources:
            text = self.extractor.extract(resource.value)
            if not text:
                empty_documents += 1
                text = self.empty_document_text
            extracted.append(
                RetrievalResource(
                    id=resource.id,
                    value=TextInput(text=text, language=resource.value.language),
                )
            )
        session = self.text_solution.prepare(extracted)
        return PdfTextExtractionSession(
            session=session,
            extractor_id=self.extractor.id,
            documents=len(extracted),
            empty_documents=empty_documents,
        )

    def release(self) -> None:
        self.text_solution.release()


class PdfTextExtractionSession:
    def __init__(
        self,
        *,
        session: Any,
        extractor_id: str,
        documents: int,
        empty_documents: int,
    ) -> None:
        self.session = session
        self.extractor_id = extractor_id
        self.documents = documents
        self.empty_documents = empty_documents

    def search_batch(
        self,
        queries: Sequence[Any],
        *,
        top_k: int,
    ) -> list[list[SearchHit]]:
        return self.session.search_batch(queries, top_k=top_k)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "pdf_extractor": self.extractor_id,
            "documents": self.documents,
            "empty_documents": self.empty_documents,
            "retriever": dict(self.session.metadata),
        }

    def close(self) -> None:
        self.session.close()
