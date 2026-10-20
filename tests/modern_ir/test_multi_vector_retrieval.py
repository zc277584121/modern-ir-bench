from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from modern_ir_bench.retrieval import ChunkedMultiVectorRetrievalSolution, RetrievalResource
from modern_ir_bench.retrieval.types import SearchHit


class LookupMultiVectorEmbedding:
    dimension = 2

    def _encode(self, inputs: Sequence[str]) -> list[np.ndarray]:
        vectors = {
            "north": [[1.0, 0.0]],
            "north east": [[1.0, 0.0], [0.0, 1.0]],
            "east": [[0.0, 1.0]],
        }
        return [np.asarray(vectors[value], dtype=np.float32) for value in inputs]

    def encode_documents(self, inputs: Sequence[str]) -> list[np.ndarray]:
        return self._encode(inputs)

    def encode_queries(self, inputs: Sequence[str]) -> list[np.ndarray]:
        return self._encode(inputs)


class ExactMaxSimIndex:
    def open(self, *, dimension: int) -> ExactMaxSimSession:
        return ExactMaxSimSession(dimension)


class ExactMaxSimSession:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors: dict[str, np.ndarray] = {}

    def add(self, ids: Sequence[str], vectors: Sequence[np.ndarray]) -> None:
        self.vectors.update(zip(ids, vectors, strict=True))

    def seal(self) -> None:
        pass

    def search(self, query_vectors: Sequence[np.ndarray], *, top_k: int) -> list[list[SearchHit]]:
        output = []
        for query in query_vectors:
            scores = [
                SearchHit(id=item_id, score=float((query @ document.T).max(axis=1).sum()))
                for item_id, document in self.vectors.items()
            ]
            output.append(sorted(scores, key=lambda hit: hit.score, reverse=True)[:top_k])
        return output

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"backend": "exact-maxsim"}

    def close(self) -> None:
        pass


def test_chunked_multi_vector_solution_collapses_hits_to_documents() -> None:
    solution = ChunkedMultiVectorRetrievalSolution(
        id="maxsim",
        title="MaxSim",
        embedding=LookupMultiVectorEmbedding(),
        index=ExactMaxSimIndex(),
        chunker=lambda text: text.split("|"),
        chunk_batch_size=2,
        candidate_multiplier=1,
    )
    session = solution.prepare(
        [
            RetrievalResource(id="d1", value="north|north"),
            RetrievalResource(id="d2", value="north east"),
            RetrievalResource(id="d3", value="east"),
        ]
    )
    try:
        ranking = session.search_batch(["north east"], top_k=3)[0]
    finally:
        session.close()

    assert [hit.id for hit in ranking] == ["d2", "d1", "d3"]
    assert session.metadata["chunks"] == 4
