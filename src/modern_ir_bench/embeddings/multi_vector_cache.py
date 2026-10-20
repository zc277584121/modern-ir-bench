"""Persistent cache for variable-length multi-vector embeddings."""

from __future__ import annotations

import sqlite3
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


class SqliteMultiVectorCache:
    """Store float32 token matrices without tying the cache to one model."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def read(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        if self.path is None or not self.path.exists() or not keys:
            return {}
        found: dict[str, np.ndarray] = {}
        with sqlite3.connect(self.path) as connection:
            self._create(connection)
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                rows = connection.execute(
                    f"SELECT cache_key, rows, dimension, vectors FROM multi_vectors "
                    f"WHERE cache_key IN ({','.join('?' for _ in batch)})",
                    list(batch),
                )
                for key, row_count, dimension, blob in rows:
                    matrix = np.frombuffer(zlib.decompress(blob), dtype="<f4").copy()
                    found[key] = matrix.reshape(row_count, dimension)
        return found

    def write(self, values: Mapping[str, np.ndarray]) -> None:
        if self.path is None or not values:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, value in values.items():
            matrix = np.asarray(value, dtype="<f4")
            if matrix.ndim != 2:
                raise ValueError("cached multi-vector values must be two-dimensional")
            rows.append(
                (
                    key,
                    matrix.shape[0],
                    matrix.shape[1],
                    zlib.compress(matrix.tobytes(order="C"), level=1),
                )
            )
        with sqlite3.connect(self.path) as connection:
            self._create(connection)
            connection.executemany(
                "INSERT OR REPLACE INTO multi_vectors(cache_key, rows, dimension, vectors) VALUES (?, ?, ?, ?)",
                rows,
            )

    @staticmethod
    def _create(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS multi_vectors ("
            "cache_key TEXT PRIMARY KEY, rows INTEGER NOT NULL, "
            "dimension INTEGER NOT NULL, vectors BLOB NOT NULL)"
        )
