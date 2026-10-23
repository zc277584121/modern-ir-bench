"""Language eligibility shared by benchmark catalogs, runners, and releases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

BENCHMARK_LANGUAGES = ("en", "zh")


class LanguageSpec(Protocol):
    id: str
    languages: tuple[str, ...]


def validate_languages(languages: Sequence[str]) -> None:
    if not languages or len(set(languages)) != len(languages) or not set(languages) <= set(BENCHMARK_LANGUAGES):
        raise ValueError(f"Invalid Solution language coverage: {languages!r}")


def scope_languages(language: str | None) -> tuple[str, ...]:
    if language is None:
        return BENCHMARK_LANGUAGES
    validate_languages((language,))
    return (language,)


def validate_solution_languages(
    solution_ids: Sequence[str],
    coverage: Mapping[str, Sequence[str]],
    required: Sequence[str],
) -> None:
    validate_languages(required)
    for solution_id in solution_ids:
        if solution_id not in coverage:
            raise ValueError(f"Unknown Solution: {solution_id}")
        supported = coverage[solution_id]
        validate_languages(supported)
        if not set(required) <= set(supported):
            raise ValueError(
                f"{solution_id} supports {', '.join(supported)}; requested scope requires {', '.join(required)}"
            )


def select_solutions(
    specs: Sequence[LanguageSpec], solution_ids: Sequence[str] | None, language: str | None
) -> list[str]:
    required = scope_languages(language)
    coverage = {spec.id: spec.languages for spec in specs}
    selected = (
        list(solution_ids)
        if solution_ids is not None
        else [spec.id for spec in specs if set(required) <= set(spec.languages)]
    )
    if len(selected) != len(set(selected)):
        raise ValueError("Solution ids must be unique")
    validate_solution_languages(selected, coverage, required)
    return selected
