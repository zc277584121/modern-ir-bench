from pathlib import Path

import pytest

from benchmarks.solution_docs import load_solution_readme


def test_load_solution_readme_uses_heading_and_first_paragraph(tmp_path: Path) -> None:
    readme = tmp_path / "solution.md"
    readme.write_text(
        "# Example `Solution`\n\n"
        "The first line uses [Milvus](https://milvus.io).\n"
        "It continues on the next line.\n\n"
        "This second paragraph is not part of the preview.\n",
        encoding="utf-8",
    )

    parsed = load_solution_readme(tmp_path, "solution.md")

    assert parsed.title == "Example Solution"
    assert parsed.preview == "The first line uses Milvus. It continues on the next line."


def test_load_solution_readme_rejects_missing_intro(tmp_path: Path) -> None:
    (tmp_path / "solution.md").write_text("# Example\n", encoding="utf-8")

    with pytest.raises(ValueError, match="opening paragraph"):
        load_solution_readme(tmp_path, "solution.md")
