"""Encode persistent embeddings before allocating a retrieval index."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from itertools import islice
from time import perf_counter
from typing import Any

from modern_ir_bench.embeddings.cached_multi_vector import CachedMultiVectorEmbedding

logger = logging.getLogger(__name__)


def pre_encode(
    embedding: CachedMultiVectorEmbedding,
    *,
    documents: Iterable[Any],
    queries: Iterable[Any],
) -> dict[str, Any]:
    """Populate both roles, release weights, and leave the encoder in cache-only mode."""
    if embedding.cache.path is None:
        raise ValueError("Pre-encoding requires a persistent embedding cache")
    counts: dict[str, Any] = {"cache": str(embedding.cache.path), "fingerprint": embedding.cache_fingerprint}
    embedding.cache_only = False
    try:
        for role, inputs, method in (
            ("documents", documents, embedding.encode_documents),
            ("queries", queries, embedding.encode_queries),
        ):
            started = perf_counter()
            iterator = iter(inputs)
            count = 0
            batches = 0
            while batch := list(islice(iterator, embedding.batch_size)):
                method(batch)
                count += len(batch)
                batches += 1
                if batches == 1:
                    counts[f"{role}_first_batch_seconds"] = perf_counter() - started
                if count % 100 < len(batch):
                    logger.info("Prepared %d %s in %.1f seconds", count, role, perf_counter() - started)
            counts[role] = count
            counts[f"{role}_batches"] = batches
            counts[f"{role}_seconds"] = perf_counter() - started
        return counts
    finally:
        embedding.release()
        embedding.cache_only = True
