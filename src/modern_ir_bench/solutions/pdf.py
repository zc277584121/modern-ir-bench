"""Composable OCR-to-retrieval Solution for PDF resources."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.retrieval import PdfInput, RetrievalResource, SearchHit, TextInput

logger = logging.getLogger(__name__)


class PdfOcrExtractor(Protocol):
    """Convert one PDF into searchable text."""

    @property
    def id(self) -> str: ...

    def extract(self, value: PdfInput) -> str: ...


class OcrPipeline(Protocol):
    """Small portion of the PaddleOCR pipeline used by the benchmark."""

    def predict_iter(self, input: str, **kwargs: Any) -> Iterable[Any]: ...


@dataclass
class PaddleOcrExtractor:
    """Render every PDF page and recognize it with bilingual mobile OCR models."""

    device: str = "cpu"
    cache_dir: Path | None = None
    detection_model: str = "PP-OCRv5_mobile_det"
    recognition_model: str = "PP-OCRv5_mobile_rec"
    recognition_score_threshold: float = 0.0
    _pipeline: OcrPipeline | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)

    @property
    def id(self) -> str:
        return "paddleocr-pp-ocrv5-mobile-all-pages-v1"

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
        cached = self._cache_path(path)
        if cached is not None and cached.is_file():
            return cached.read_text(encoding="utf-8")

        pages = [self._page_text(result) for result in self._get_pipeline().predict_iter(input=str(path))]
        text = "\n\n".join(page for page in pages if page).strip()
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_suffix(".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(cached)
        return text

    def _get_pipeline(self) -> OcrPipeline:
        if self._pipeline is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as error:
                raise RuntimeError(
                    "Install OCR dependencies with `uv sync --group pdf --group pdf-cpu` "
                    "or `uv sync --group pdf --group pdf-gpu`"
                ) from error
            self._pipeline = cast(
                OcrPipeline,
                PaddleOCR(
                    text_detection_model_name=self.detection_model,
                    text_recognition_model_name=self.recognition_model,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device=self.device,
                ),
            )
        return self._pipeline

    def _cache_path(self, path: Path) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return self.cache_dir / f"{digest.hexdigest()}.txt"

    def _page_text(self, result: Any) -> str:
        payload = getattr(result, "json", result)
        if callable(payload):
            payload = payload()
        if not isinstance(payload, Mapping):
            raise TypeError(f"Unexpected PaddleOCR result: {type(payload).__name__}")
        if isinstance(payload.get("res"), Mapping):
            payload = payload["res"]
        texts = payload.get("rec_texts")
        scores = payload.get("rec_scores")
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise TypeError("PaddleOCR result does not contain rec_texts")
        if scores is None:
            return "\n".join(str(text).strip() for text in texts if str(text).strip())
        return "\n".join(
            str(text).strip()
            for text, score in zip(texts, scores, strict=True)
            if str(text).strip() and float(score) >= self.recognition_score_threshold
        )


@dataclass(frozen=True, kw_only=True)
class PdfOcrRetrievalSolution(Solution):
    """OCR complete PDFs, then delegate indexing and search to a text Solution."""

    text_solution: Solution
    extractor: PdfOcrExtractor
    empty_document_text: str = "[no extractable PDF text]"
    progress_every: int = 100

    def prepare(
        self,
        resources: Iterable[RetrievalResource[PdfInput]],
    ) -> PdfOcrRetrievalSession:
        extracted: list[RetrievalResource[TextInput]] = []
        empty_documents = 0
        started_at = time.monotonic()
        for index, resource in enumerate(resources, start=1):
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
            if self.progress_every > 0 and index % self.progress_every == 0:
                elapsed = time.monotonic() - started_at
                logger.info(
                    "OCR processed %d PDFs in %.1f minutes (%.2f PDFs/s)",
                    index,
                    elapsed / 60,
                    index / elapsed,
                )
        session = self.text_solution.prepare(extracted)
        return PdfOcrRetrievalSession(
            session=session,
            extractor_id=self.extractor.id,
            documents=len(extracted),
            empty_documents=empty_documents,
        )

    def release(self) -> None:
        self.text_solution.release()


class PdfOcrRetrievalSession:
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
