"""Composable late-interaction retrieval Solutions."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.embeddings.protocols import MultiVectorEmbedding
from modern_ir_bench.retrieval._chunk_ranking import (
    candidate_depths,
    collapse_chunk_rankings,
    complete_document_rankings,
    has_enough_documents,
)
from modern_ir_bench.retrieval.dense import _batches
from modern_ir_bench.retrieval.indexes.protocols import MultiVectorIndex, MultiVectorIndexSession
from modern_ir_bench.retrieval.types import RetrievalResource, SearchHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class MultiVectorRetrievalSolution(Solution):
    """Index one token matrix per resource and rank with MaxSim."""

    embedding: MultiVectorEmbedding
    index: MultiVectorIndex
    document_batch_size: int = 16
    document_input: Callable[[Any], Any] = lambda value: value
    query_input: Callable[[Any], Any] = lambda value: value

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.document_batch_size < 1:
            raise ValueError("document_batch_size must be positive")

    def prepare(self, resources: Iterable[RetrievalResource[Any]]) -> MultiVectorRetrievalSession:
        index_session = self.index.open(dimension=self.embedding.dimension)
        try:
            count = 0
            for batch in _batches(resources, self.document_batch_size):
                vectors = self.embedding.encode_documents(
                    [self.document_input(resource.value) for resource in batch]
                )
                index_session.add([resource.id for resource in batch], vectors)
                count += len(batch)
            if count == 0:
                raise ValueError("A multi-vector retrieval Solution requires at least one resource")
            index_session.seal()
        except Exception:
            index_session.close()
            raise
        return MultiVectorRetrievalSession(
            embedding=self.embedding,
            index=index_session,
            query_input=self.query_input,
        )

    def release(self) -> None:
        release = getattr(self.embedding, "release", None)
        if release is not None:
            release()


class MultiVectorRetrievalSession:
    def __init__(
        self,
        *,
        embedding: MultiVectorEmbedding,
        index: MultiVectorIndexSession,
        query_input: Callable[[Any], Any],
    ) -> None:
        self.embedding = embedding
        self.index = index
        self.query_input = query_input

    def search_batch(self, queries: Sequence[Any], *, top_k: int) -> list[list[SearchHit]]:
        vectors = self.embedding.encode_queries([self.query_input(query) for query in queries])
        return self.index.search(vectors, top_k=top_k)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.index.metadata

    def close(self) -> None:
        self.index.close()


@dataclass(frozen=True, kw_only=True)
class ChunkedMultiVectorRetrievalSolution(Solution):
    """Run MaxSim over chunks or pages and collapse hits to resources."""

    embedding: MultiVectorEmbedding
    index: MultiVectorIndex
    chunker: Callable[[Any], Sequence[Any]]
    chunk_batch_size: int = 16
    candidate_multiplier: int = 4
    progress_every: int = 100
    document_input: Callable[[Any], Any] = lambda value: value
    query_input: Callable[[Any], Any] = lambda value: value

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.chunk_batch_size < 1:
            raise ValueError("chunk_batch_size must be positive")
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        if self.progress_every < 0:
            raise ValueError("progress_every cannot be negative")

    def pre_encode(self, resources: Iterable[RetrievalResource[Any]], queries: Iterable[Any]) -> dict[str, Any]:
        """Persist page/chunk and query embeddings without opening an index."""
        from modern_ir_bench.embeddings.cached_multi_vector import CachedMultiVectorEmbedding
        from modern_ir_bench.embeddings.preparation import pre_encode

        if not isinstance(self.embedding, CachedMultiVectorEmbedding):
            raise TypeError("Pre-encoding requires a persistent multi-vector encoder")

        def documents() -> Iterable[Any]:
            for resource in resources:
                chunks = self.chunker(self.document_input(resource.value))
                if not chunks:
                    raise ValueError(f"chunker returned no chunks for resource {resource.id}")
                yield from chunks

        return pre_encode(
            self.embedding,
            documents=documents(),
            queries=(self.query_input(query) for query in queries),
        )

    def prepare(self, resources: Iterable[RetrievalResource[Any]]) -> ChunkedMultiVectorRetrievalSession:
        index_session = self.index.open(dimension=self.embedding.dimension)
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
            started_at = time.monotonic()
            chunk_count = 0
            for batch in _batches(chunks(), self.chunk_batch_size):
                vectors = self.embedding.encode_documents([chunk.value for chunk in batch])
                index_session.add([chunk.id for chunk in batch], vectors)
                chunk_count += len(batch)
                if self.progress_every > 0 and chunk_count // self.progress_every != (
                    chunk_count - len(batch)
                ) // self.progress_every:
                    elapsed = time.monotonic() - started_at
                    logger.info(
                        "MaxSim encoded and indexed %d units in %.1f minutes (%.2f units/s)",
                        chunk_count,
                        elapsed / 60,
                        chunk_count / elapsed,
                    )
            if not chunk_to_document:
                raise ValueError("A chunked multi-vector retrieval Solution requires at least one chunk")
            index_session.seal()
        except Exception:
            index_session.close()
            raise
        return ChunkedMultiVectorRetrievalSession(
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


class ChunkedMultiVectorRetrievalSession(MultiVectorRetrievalSession):
    def __init__(
        self,
        *,
        embedding: MultiVectorEmbedding,
        index: MultiVectorIndexSession,
        chunk_to_document: Mapping[str, str],
        candidate_multiplier: int,
        query_input: Callable[[Any], Any],
    ) -> None:
        super().__init__(embedding=embedding, index=index, query_input=query_input)
        self.chunk_to_document = dict(chunk_to_document)
        self.candidate_multiplier = candidate_multiplier

    def search_batch(self, queries: Sequence[Any], *, top_k: int) -> list[list[SearchHit]]:
        vectors = self.embedding.encode_queries([self.query_input(query) for query in queries])
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
            if has_enough_documents(document_rankings, document_count=document_count, top_k=top_k):
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
            "backend": "chunked-multi-vector-retrieval",
            "chunks": len(self.chunk_to_document),
            "candidate_multiplier": self.candidate_multiplier,
            "index": dict(self.index.metadata),
        }
