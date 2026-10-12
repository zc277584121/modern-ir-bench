from __future__ import annotations

from benchmarks.pdf_retrieval.solutions import SOLUTION_SPECS


def test_pdf_solution_catalog_has_unique_complete_entries() -> None:
    ids = [spec.id for spec in SOLUTION_SPECS]

    assert len(ids) == len(set(ids))
    assert "paddleocr-v5-mobile-bm25-voyage-multimodal-3-5-rrf" in ids
    assert all(spec.title and spec.description and spec.tags and spec.readme_path for spec in SOLUTION_SPECS)
