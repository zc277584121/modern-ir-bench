from __future__ import annotations

import numpy as np

from modern_ir_bench.embeddings.multi_vector_cache import SqliteMultiVectorCache


def test_multi_vector_cache_round_trip(tmp_path) -> None:
    cache = SqliteMultiVectorCache(tmp_path / "vectors.sqlite3")
    expected = {
        "short": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "long": np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    }

    cache.write(expected)
    actual = cache.read(["missing", "short", "long"])

    assert set(actual) == {"short", "long"}
    np.testing.assert_array_equal(actual["short"], expected["short"])
    np.testing.assert_array_equal(actual["long"], expected["long"])
