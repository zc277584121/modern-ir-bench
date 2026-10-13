"""Durable query-document scores for reranking experiments."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path


class SqliteRerankCache:
    """Store exact reranker scores while invalidating changed text automatically."""

    def __init__(self, path: Path, *, namespace: str) -> None:
        if not namespace:
            raise ValueError("Rerank cache namespace must be non-empty")
        self.path = Path(path)
        self.namespace = namespace

    def read(
        self,
        *,
        reranker_id: str,
        query: str,
        documents: Sequence[tuple[str, str]],
    ) -> dict[str, float]:
        if not self.path.is_file() or not documents:
            return {}
        query_hash = self._hash(query)
        found: dict[str, float] = {}
        with sqlite3.connect(self.path, timeout=30) as connection:
            self._create(connection)
            for document_id, text in documents:
                row = connection.execute(
                    "SELECT score FROM rerank_scores "
                    "WHERE namespace = ? AND reranker_id = ? AND query_hash = ? "
                    "AND document_id = ? AND document_hash = ?",
                    (self.namespace, reranker_id, query_hash, document_id, self._hash(text)),
                ).fetchone()
                if row is not None:
                    found[document_id] = float(row[0])
        return found

    def write(
        self,
        *,
        reranker_id: str,
        query: str,
        documents: Sequence[tuple[str, str]],
        scores: Mapping[str, float],
    ) -> None:
        if set(scores) != {document_id for document_id, _ in documents}:
            raise ValueError("Rerank scores must cover every supplied document exactly once")
        if not documents:
            return
        query_hash = self._hash(query)
        rows = [
            (
                self.namespace,
                reranker_id,
                query_hash,
                document_id,
                self._hash(text),
                float(scores[document_id]),
            )
            for document_id, text in documents
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=30) as connection:
            self._create(connection)
            connection.executemany(
                "INSERT OR REPLACE INTO rerank_scores"
                "(namespace, reranker_id, query_hash, document_id, document_hash, score) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _create(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS rerank_scores ("
            "namespace TEXT NOT NULL, "
            "reranker_id TEXT NOT NULL, "
            "query_hash TEXT NOT NULL, "
            "document_id TEXT NOT NULL, "
            "document_hash TEXT NOT NULL, "
            "score REAL NOT NULL, "
            "PRIMARY KEY(namespace, reranker_id, query_hash, document_id, document_hash)"
            ")"
        )
