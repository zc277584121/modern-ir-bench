"""Composable chunked learned-sparse retrieval Solution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.embeddings.sparse import SparseEmbedding
from modern_ir_bench.retrieval._chunk_ranking import (
    candidate_depths,
    collapse_chunk_rankings,
    complete_document_rankings,
    has_enough_documents,
)
from modern_ir_bench.retrieval.dense import _batches
from modern_ir_bench.retrieval.indexes.sparse import SparseIndex, SparseIndexSession
from modern_ir_bench.retrieval.types import RetrievalResource, SearchHit


@dataclass(frozen=True, kw_only=True)
class ChunkedSparseRetrievalSolution(Solution):
    embedding: SparseEmbedding
    index: SparseIndex
    chunker: Callable[[Any], Sequence[Any]]
    chunk_batch_size: int = 64
    candidate_multiplier: int = 4
    document_input: Callable[[Any], Any] = lambda value: value
    query_input: Callable[[Any], Any] = lambda value: value

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.chunk_batch_size < 1 or self.candidate_multiplier < 1:
            raise ValueError("batch size and candidate multiplier must be positive")

    def prepare(
        self,
        resources: Iterable[RetrievalResource[Any]],
    ) -> ChunkedSparseRetrievalSession:
        index_session = self.index.open()
        chunk_to_document: dict[str, str] = {}

        def chunks() -> Iterable[RetrievalResource[Any]]:
            for resource in resources:
                values = list(self.chunker(self.document_input(resource.value)))
                if not values:
                    raise ValueError(f"chunker returned no chunks for resource {resource.id}")
                for chunk_index, value in enumerate(values):
                    chunk_id = f"{resource.id}:chunk:{chunk_index:04d}"
                    if chunk_id in chunk_to_document:
                        raise ValueError(f"duplicate chunk id: {chunk_id}")
                    chunk_to_document[chunk_id] = resource.id
                    yield RetrievalResource(id=chunk_id, value=value)

        try:
            for batch in _batches(chunks(), self.chunk_batch_size):
                index_session.add(
                    [chunk.id for chunk in batch],
                    self.embedding.encode_sparse_documents([chunk.value for chunk in batch]),
                )
            if not chunk_to_document:
                raise ValueError("A chunked sparse Solution requires at least one chunk")
            index_session.seal()
        except Exception:
            index_session.close()
            raise
        return ChunkedSparseRetrievalSession(
            embedding=self.embedding,
            index=index_session,
            chunk_to_document=chunk_to_document,
            candidate_multiplier=self.candidate_multiplier,
            query_input=self.query_input,
        )

    def release(self) -> None:
        release = getattr(self.embedding, "release", None)
        if release is not None:
            release()


class ChunkedSparseRetrievalSession:
    def __init__(
        self,
        *,
        embedding: SparseEmbedding,
        index: SparseIndexSession,
        chunk_to_document: Mapping[str, str],
        candidate_multiplier: int,
        query_input: Callable[[Any], Any],
    ) -> None:
        self.embedding = embedding
        self.index = index
        self.chunk_to_document = dict(chunk_to_document)
        self.candidate_multiplier = candidate_multiplier
        self.query_input = query_input

    def search_batch(
        self,
        queries: Sequence[Any],
        *,
        top_k: int,
    ) -> list[list[SearchHit]]:
        vectors = self.embedding.encode_sparse_queries([self.query_input(query) for query in queries])
        document_count = len(set(self.chunk_to_document.values()))
        for candidate_depth in candidate_depths(
            unit_count=len(self.chunk_to_document),
            top_k=top_k,
            initial_multiplier=self.candidate_multiplier,
        ):
            document_rankings = collapse_chunk_rankings(
                self.index.search(vectors, top_k=candidate_depth),
                unit_to_document=self.chunk_to_document,
                top_k=top_k,
            )
            if has_enough_documents(
                document_rankings,
                document_count=document_count,
                top_k=top_k,
            ):
                return document_rankings
            if candidate_depth == len(self.chunk_to_document):
                return complete_document_rankings(
                    document_rankings,
                    document_ids=self.chunk_to_document.values(),
                    top_k=top_k,
                )
        raise AssertionError("candidate depth iteration must return")

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "chunked-sparse-retrieval",
            "chunks": len(self.chunk_to_document),
            "candidate_multiplier": self.candidate_multiplier,
            "index": dict(self.index.metadata),
        }

    def close(self) -> None:
        self.index.close()
