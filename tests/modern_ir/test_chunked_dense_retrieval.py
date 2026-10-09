from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from modern_ir_bench.retrieval import ChunkedDenseRetrievalSolution, RetrievalResource
from modern_ir_bench.retrieval.indexes import NumpyFlatIndex


class LookupEmbedding:
    dimension = 2

    def _encode(self, inputs: Sequence[str]) -> np.ndarray:
        vectors = {
            "north": [1.0, 0.0],
            "north east": [0.8, 0.2],
            "east": [0.0, 1.0],
        }
        return np.asarray([vectors[value] for value in inputs], dtype=np.float32)

    def encode_documents(self, inputs: Sequence[str]) -> np.ndarray:
        return self._encode(inputs)

    def encode_queries(self, inputs: Sequence[str]) -> np.ndarray:
        return self._encode(inputs)


def test_chunked_dense_solution_collapses_multiple_chunk_hits_to_documents() -> None:
    solution = ChunkedDenseRetrievalSolution(
        id="chunked",
        title="Chunked",
        embedding=LookupEmbedding(),
        index=NumpyFlatIndex(),
        chunker=lambda text: text.split("|"),
        chunk_batch_size=2,
        candidate_multiplier=3,
    )
    session = solution.prepare(
        [
            RetrievalResource(id="d1", value="north|north east"),
            RetrievalResource(id="d2", value="east"),
        ]
    )
    try:
        ranking = session.search_batch(["north"], top_k=2)[0]
    finally:
        session.close()

    assert [hit.id for hit in ranking] == ["d1", "d2"]
    assert session.metadata["chunks"] == 3
