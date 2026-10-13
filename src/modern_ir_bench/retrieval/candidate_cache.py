"""Durable component rankings for multi-retriever fusion."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from modern_ir_bench.retrieval.types import RetrievalResource, SearchHit


class SqliteCandidateCache:
    """Store document-level candidate lists returned by one retrieval branch."""

    def __init__(self, path: Path, *, namespace: str) -> None:
        if not namespace:
            raise ValueError("Candidate cache namespace must be non-empty")
        self.path = Path(path)
        self.namespace = namespace

    def read(
        self,
        *,
        corpus_key: str,
        component_id: str,
        queries: Sequence[Any],
        depth: int,
    ) -> dict[int, list[SearchHit]]:
        if not self.path.is_file() or not queries:
            return {}
        found: dict[int, list[SearchHit]] = {}
        with sqlite3.connect(self.path, timeout=30) as connection:
            self._create(connection)
            for index, query in enumerate(queries):
                row = connection.execute(
                    "SELECT hits FROM candidate_lists "
                    "WHERE namespace = ? AND corpus_key = ? AND component_id = ? "
                    "AND query_key = ? AND depth >= ? "
                    "ORDER BY depth ASC LIMIT 1",
                    (self.namespace, corpus_key, component_id, self._query_key(query), depth),
                ).fetchone()
                if row is None:
                    continue
                payload = json.loads(row[0])
                found[index] = [
                    SearchHit(id=str(document_id), score=float(score)) for document_id, score in payload[:depth]
                ]
        return found

    def write(
        self,
        *,
        corpus_key: str,
        component_id: str,
        queries: Sequence[Any],
        depth: int,
        rankings: Sequence[Sequence[SearchHit]],
    ) -> None:
        if len(queries) != len(rankings):
            raise ValueError("Queries and candidate rankings must contain the same number of rows")
        if not queries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            (
                self.namespace,
                corpus_key,
                component_id,
                self._query_key(query),
                depth,
                json.dumps(
                    [[hit.id, float(hit.score)] for hit in ranking],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            for query, ranking in zip(queries, rankings, strict=True)
        ]
        with sqlite3.connect(self.path, timeout=30) as connection:
            self._create(connection)
            connection.executemany(
                "INSERT OR REPLACE INTO candidate_lists"
                "(namespace, corpus_key, component_id, query_key, depth, hits) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    @staticmethod
    def corpus_key(resources: Sequence[RetrievalResource[Any]]) -> str:
        digest = hashlib.sha256()
        for resource in resources:
            encoded = resource.id.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def _query_key(query: Any) -> str:
        if is_dataclass(query) and not isinstance(query, type):
            value: Any = {
                "type": f"{type(query).__module__}.{type(query).__qualname__}",
                "value": asdict(query),
            }
        elif isinstance(query, Path):
            value = {"type": "path", "value": str(query)}
        else:
            value = {"type": f"{type(query).__module__}.{type(query).__qualname__}", "value": query}
        try:
            encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        except TypeError:
            encoded = repr(value).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _create(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS candidate_lists ("
            "namespace TEXT NOT NULL, "
            "corpus_key TEXT NOT NULL, "
            "component_id TEXT NOT NULL, "
            "query_key TEXT NOT NULL, "
            "depth INTEGER NOT NULL, "
            "hits TEXT NOT NULL, "
            "PRIMARY KEY(namespace, corpus_key, component_id, query_key, depth)"
            ")"
        )
