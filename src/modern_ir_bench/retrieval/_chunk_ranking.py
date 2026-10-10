"""Shared helpers for collapsing chunk hits into document rankings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from modern_ir_bench.retrieval.types import SearchHit


def candidate_depths(
    *,
    unit_count: int,
    top_k: int,
    initial_multiplier: int,
) -> Iterable[int]:
    """Increase candidate depth until enough documents are found or all units are searched."""

    if unit_count < 1 or top_k < 1 or initial_multiplier < 1:
        raise ValueError("unit count, top_k, and initial multiplier must be positive")
    depth = min(unit_count, max(top_k, top_k * initial_multiplier))
    while True:
        yield depth
        if depth == unit_count:
            return
        depth = min(unit_count, depth * 2)


def collapse_chunk_rankings(
    rankings: Sequence[Sequence[SearchHit]],
    *,
    unit_to_document: Mapping[str, str],
    top_k: int,
) -> list[list[SearchHit]]:
    output: list[list[SearchHit]] = []
    for unit_hits in rankings:
        seen: set[str] = set()
        documents: list[SearchHit] = []
        for hit in unit_hits:
            document_id = unit_to_document[hit.id]
            if document_id in seen:
                continue
            seen.add(document_id)
            documents.append(SearchHit(id=document_id, score=hit.score))
            if len(documents) == top_k:
                break
        output.append(documents)
    return output


def has_enough_documents(
    rankings: Sequence[Sequence[SearchHit]],
    *,
    document_count: int,
    top_k: int,
) -> bool:
    expected = min(document_count, top_k)
    return all(len(ranking) == expected for ranking in rankings)


def complete_document_rankings(
    rankings: Sequence[Sequence[SearchHit]],
    *,
    document_ids: Iterable[str],
    top_k: int,
) -> list[list[SearchHit]]:
    """Deterministically append zero-score documents when a sparse search omits ties."""

    ordered_ids = sorted(set(document_ids))
    output: list[list[SearchHit]] = []
    for ranking in rankings:
        completed = list(ranking)
        seen = {hit.id for hit in completed}
        completed.extend(SearchHit(id=document_id, score=0.0) for document_id in ordered_ids if document_id not in seen)
        output.append(completed[:top_k])
    return output
