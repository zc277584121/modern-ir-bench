"""Composable OCR-to-retrieval Solution for PDF resources."""

from __future__ import annotations

import hashlib
import json
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


@dataclass(frozen=True)
class PdfPageRenderer:
    """Render PDF pages to consistently sized cached JPEG images."""

    cache_dir: Path
    scale: float = 2.0
    max_side: int = 1600
    jpeg_quality: int = 88

    def __post_init__(self) -> None:
        if self.scale <= 0 or self.max_side < 1:
            raise ValueError("scale and max_side must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))

    @property
    def id(self) -> str:
        return f"pdfium-scale-{self.scale:g}-max-{self.max_side}-jpeg-{self.jpeg_quality}"

    def render(self, value: PdfInput) -> list[Path]:
        digest = self._pdf_digest(value)
        renderer_digest = hashlib.sha256(self.id.encode()).hexdigest()[:12]
        output_dir = self.cache_dir / renderer_digest / digest
        manifest_path = output_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pages = [output_dir / name for name in manifest["pages"]]
            if manifest.get("renderer") == self.id and pages and all(page.is_file() for page in pages):
                return pages

        import pypdfium2 as pdfium
        from PIL import Image

        source: str | bytes
        if value.path and Path(value.path).is_file():
            source = str(Path(value.path))
        elif value.data is not None:
            source = value.data
        else:
            raise FileNotFoundError(f"PDF path is not accessible: {value.path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        document = pdfium.PdfDocument(source)
        page_names = []
        try:
            for page_index in range(len(document)):
                page_name = f"page-{page_index + 1:04d}.jpg"
                page_path = output_dir / page_name
                if not page_path.is_file():
                    page = document[page_index]
                    try:
                        image = page.render(scale=self.scale).to_pil().convert("RGB")
                        if max(image.size) > self.max_side:
                            image.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
                        temporary = page_path.with_suffix(".tmp.jpg")
                        image.save(temporary, format="JPEG", quality=self.jpeg_quality, optimize=True)
                        temporary.replace(page_path)
                    finally:
                        page.close()
                page_names.append(page_name)
        finally:
            document.close()
        temporary_manifest = manifest_path.with_suffix(".tmp")
        temporary_manifest.write_text(
            json.dumps({"renderer": self.id, "pages": page_names}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
        return [output_dir / name for name in page_names]

    @staticmethod
    def _pdf_digest(value: PdfInput) -> str:
        digest = hashlib.sha256()
        if value.path and Path(value.path).is_file():
            with Path(value.path).open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
        elif value.data is not None:
            digest.update(value.data)
        else:
            raise FileNotFoundError(f"PDF path is not accessible: {value.path}")
        return digest.hexdigest()


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
