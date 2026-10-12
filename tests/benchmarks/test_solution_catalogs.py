from __future__ import annotations

import pytest

from benchmarks.pdf_retrieval.solutions import SOLUTION_SPECS as PDF_SOLUTION_SPECS
from benchmarks.ranked_retrieval.solutions import SOLUTION_SPECS as RANKED_SOLUTION_SPECS


@pytest.mark.parametrize("specs", [RANKED_SOLUTION_SPECS, PDF_SOLUTION_SPECS])
def test_solution_catalog_has_unique_complete_entries(specs) -> None:
    ids = [spec.id for spec in specs]

    assert len(ids) == len(set(ids))
    assert all(spec.title and spec.description and spec.tags and spec.readme_path for spec in specs)
