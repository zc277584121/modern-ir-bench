"""Small persistent cache shared by deterministic embedding APIs."""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path


class SqliteVectorCache:
    def __init__(self, path: Path | None) -> None:
        self.path = path

    def read(self, keys: Sequence[str]) -> dict[str, list[float]]:
        if self.path is None or not self.path.exists() or not keys:
            return {}
        found: dict[str, list[float]] = {}
        with sqlite3.connect(self.path) as connection:
            self._create(connection)
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                rows = connection.execute(
                    f"SELECT cache_key, dimension, vector FROM embeddings "
                    f"WHERE cache_key IN ({','.join('?' for _ in batch)})",
                    list(batch),
                )
                found.update({key: list(struct.unpack(f"<{dimension}f", blob)) for key, dimension, blob in rows})
        return found

    def write(self, values: Mapping[str, Sequence[float]]) -> None:
        if self.path is None or not values:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            (key, len(vector), struct.pack(f"<{len(vector)}f", *vector))
            for key, vector in values.items()
        ]
        with sqlite3.connect(self.path) as connection:
            self._create(connection)
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings(cache_key, dimension, vector) VALUES (?, ?, ?)",
                rows,
            )

    @staticmethod
    def _create(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, dimension INTEGER, vector BLOB)"
        )
