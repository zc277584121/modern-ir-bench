"""Read public Solution summaries from their canonical Markdown files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)]\([^)]*\)")


@dataclass(frozen=True)
class SolutionReadme:
    """The title and opening paragraph shown in the Space preview."""

    title: str
    preview: str


def _plain_text(value: str) -> str:
    value = _MARKDOWN_LINK.sub(r"\1", value)
    return value.replace("`", "").replace("**", "").replace("__", "").strip()


def load_solution_readme(repository_root: Path, readme_path: str) -> SolutionReadme:
    """Extract the H1 and first prose paragraph from a Solution README."""

    path = (repository_root / readme_path).resolve()
    try:
        path.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"Solution README must be inside the repository: {readme_path}") from error

    lines = path.read_text(encoding="utf-8").splitlines()
    title = next(
        (_plain_text(line.removeprefix("# ")) for line in lines if line.startswith("# ")),
        "",
    )
    paragraph: list[str] = []
    title_seen = False
    for line in lines:
        stripped = line.strip()
        if not title_seen:
            title_seen = line.startswith("# ")
            continue
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#"):
            if paragraph:
                break
            continue
        paragraph.append(stripped)

    preview = _plain_text(" ".join(paragraph))
    if not title or not preview:
        raise ValueError(f"Solution README needs an H1 and opening paragraph: {readme_path}")
    return SolutionReadme(title=title, preview=preview)
