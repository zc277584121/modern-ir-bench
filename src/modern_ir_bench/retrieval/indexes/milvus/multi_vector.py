"""Native Milvus MaxSim index for variable-length embedding lists."""

from __future__ import annotations

import importlib.metadata
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modern_ir_bench.embeddings.protocols import MultiVectors
from modern_ir_bench.retrieval.indexes.milvus.index import _collection_name
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget
from modern_ir_bench.retrieval.types import SearchHit


def _matrices(
    vectors: MultiVectors,
    *,
    dimension: int,
    max_vectors: int,
) -> list[np.ndarray]:
    output = []
    for value in vectors:
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != dimension:
            raise ValueError(f"Expected non-empty (*, {dimension}) token matrices, got {matrix.shape}")
        if matrix.shape[0] > max_vectors:
            raise ValueError(
                f"Multi-vector input contains {matrix.shape[0]} vectors; configured maximum is {max_vectors}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("multi-vector input contains a non-finite value")
        output.append(matrix)
    return output


@dataclass(frozen=True)
class MilvusMultiVectorIndex:
    """Create an isolated Milvus 3 collection using MAX_SIM_COSINE."""

    target: MilvusTarget
    max_vectors: int = 4096
    index_type: str = "AUTOINDEX"
    index_params: Mapping[str, Any] = field(default_factory=dict)
    search_params: Mapping[str, Any] = field(default_factory=dict)
    collection_prefix: str = "modern_ir_maxsim"
    index_build_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_vectors <= 4096:
            raise ValueError("max_vectors must be between 1 and Milvus's 4096-vector limit")
        if self.target.deployment == "milvus-lite":
            raise ValueError("Milvus Lite does not support StructArray EmbeddingList search")
        if self.index_type.upper() != "AUTOINDEX":
            raise ValueError("Milvus EmbeddingList search currently requires AUTOINDEX")
        if self.index_build_timeout_seconds <= 0:
            raise ValueError("index_build_timeout_seconds must be positive")

    def open(self, *, dimension: int) -> MilvusMultiVectorSession:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        from pymilvus import DataType, MilvusClient

        client = MilvusClient(**self.target.client_arguments())
        collection_name = _collection_name(self.collection_prefix)
        try:
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                max_length=2048,
                is_primary=True,
            )
            vector_struct = client.create_struct_field_schema()
            vector_struct.add_field(
                field_name="embedding",
                datatype=DataType.FLOAT_VECTOR,
                dim=dimension,
            )
            schema.add_field(
                field_name="vectors",
                datatype=DataType.ARRAY,
                element_type=DataType.STRUCT,
                struct_schema=vector_struct,
                max_capacity=self.max_vectors,
            )
            indexes = MilvusClient.prepare_index_params()
            indexes.add_index(
                field_name="vectors[embedding]",
                index_name="maxsim_index",
                index_type="AUTOINDEX",
                metric_type="MAX_SIM_COSINE",
                params=dict(self.index_params),
            )
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=indexes,
                consistency_level="Strong",
            )
        except Exception:
            client.close()
            raise
        return MilvusMultiVectorSession(
            client=client,
            collection_name=collection_name,
            dimension=dimension,
            max_vectors=self.max_vectors,
            search_params=self.search_params,
            deployment=self.target.deployment,
            index_build_timeout_seconds=self.index_build_timeout_seconds,
        )


class MilvusMultiVectorSession:
    def __init__(
        self,
        *,
        client: Any,
        collection_name: str,
        dimension: int,
        max_vectors: int,
        search_params: Mapping[str, Any],
        deployment: str,
        index_build_timeout_seconds: float,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension
        self.max_vectors = max_vectors
        self.search_params = dict(search_params)
        self.deployment = deployment
        self.index_build_timeout_seconds = index_build_timeout_seconds
        self._ids: set[str] = set()
        self._vector_count = 0
        self._sealed = False
        self._closed = False
        self._actual_index: Mapping[str, Any] | None = None
        self._server_version: str | None = None
        self._server_count: int | None = None

    def add(self, ids: Sequence[str], vectors: MultiVectors) -> None:
        if self._closed:
            raise RuntimeError("Cannot add vectors to a closed Milvus index")
        if self._sealed:
            raise RuntimeError("Cannot add vectors after the index has been sealed")
        matrices = _matrices(vectors, dimension=self.dimension, max_vectors=self.max_vectors)
        normalized_ids = [str(item_id) for item_id in ids]
        if len(normalized_ids) != len(matrices):
            raise ValueError("ids and vectors must contain the same number of rows")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("ids must be unique within a batch")
        duplicates = self._ids.intersection(normalized_ids)
        if duplicates:
            raise ValueError(f"Duplicate ids across batches: {sorted(duplicates)}")
        self.client.insert(
            collection_name=self.collection_name,
            data=[
                {
                    "id": item_id,
                    "vectors": [{"embedding": vector.tolist()} for vector in matrix],
                }
                for item_id, matrix in zip(normalized_ids, matrices, strict=True)
            ],
        )
        self._ids.update(normalized_ids)
        self._vector_count += sum(matrix.shape[0] for matrix in matrices)

    def seal(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot seal a closed Milvus index")
        if self._sealed:
            return
        if not self._ids:
            raise ValueError("Cannot seal an empty multi-vector index")
        self.client.flush(collection_name=self.collection_name, timeout=300)
        self.client.load_collection(collection_name=self.collection_name, timeout=300)
        deadline = time.monotonic() + self.index_build_timeout_seconds
        while True:
            self._actual_index = self.client.describe_index(
                collection_name=self.collection_name,
                index_name="maxsim_index",
            )
            state = self._actual_index.get("state")
            pending = int(self._actual_index.get("pending_index_rows", 0))
            if state == "Finished" and pending == 0:
                break
            if state == "Failed":
                raise RuntimeError(f"Milvus MaxSim index build failed: {self._actual_index}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Milvus MaxSim index did not finish within {self.index_build_timeout_seconds:g} seconds: "
                    f"{self._actual_index}"
                )
            time.sleep(0.5)
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        self._server_count = int(stats["row_count"])
        if self._server_count != len(self._ids):
            raise RuntimeError(f"Milvus reports {self._server_count} rows after inserting {len(self._ids)}")
        self._server_version = self.client.get_server_version()
        self._sealed = True

    def search(self, query_vectors: MultiVectors, *, top_k: int) -> list[list[SearchHit]]:
        if not self._sealed or self._closed:
            raise RuntimeError("The Milvus index must be sealed before search")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        from pymilvus.client.embedding_list import EmbeddingList

        matrices = _matrices(
            query_vectors,
            dimension=self.dimension,
            max_vectors=self.max_vectors,
        )
        data = []
        for matrix in matrices:
            embedding_list = EmbeddingList()
            embedding_list.add_batch(matrix.tolist())
            data.append(embedding_list)
        results = self.client.search(
            collection_name=self.collection_name,
            data=data,
            anns_field="vectors[embedding]",
            limit=min(top_k, len(self._ids)),
            search_params=dict(self.search_params),
            consistency_level="Strong",
        )
        return [
            [SearchHit(id=str(hit["id"]), score=float(hit["distance"])) for hit in query_hits]
            for query_hits in results
        ]

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "milvus-embedding-list",
            "deployment": self.deployment,
            "metric": "MAX_SIM_COSINE",
            "requested_index_type": "AUTOINDEX",
            "actual_index": dict(self._actual_index) if self._actual_index else None,
            "dimension": self.dimension,
            "count": len(self._ids),
            "token_vectors": self._vector_count,
            "max_vectors": self.max_vectors,
            "server_count": self._server_count,
            "pymilvus_version": importlib.metadata.version("pymilvus"),
            "server_version": self._server_version,
            "consistency_level": "Strong",
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.client.drop_collection(collection_name=self.collection_name)
        finally:
            self.client.close()
            self._closed = True
