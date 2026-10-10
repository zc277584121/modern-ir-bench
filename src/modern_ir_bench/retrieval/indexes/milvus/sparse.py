"""Isolated learned-sparse index sessions backed by Milvus."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modern_ir_bench.embeddings.sparse import SparseVector
from modern_ir_bench.retrieval.indexes.milvus.index import _collection_name
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget
from modern_ir_bench.retrieval.types import SearchHit


@dataclass(frozen=True)
class MilvusSparseIndex:
    target: MilvusTarget
    index_params: Mapping[str, Any] = field(default_factory=dict)
    search_params: Mapping[str, Any] = field(default_factory=dict)
    collection_prefix: str = "modern_ir_sparse"

    def open(self) -> MilvusSparseSession:
        from pymilvus import DataType, MilvusClient

        arguments = dict(self.target.client_arguments())
        if self.target.deployment == "milvus-lite":
            Path(arguments["uri"]).parent.mkdir(parents=True, exist_ok=True)
        client = MilvusClient(**arguments)
        collection_name = _collection_name(self.collection_prefix)
        try:
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, max_length=2048, is_primary=True)
            schema.add_field("vector", DataType.SPARSE_FLOAT_VECTOR)
            indexes = MilvusClient.prepare_index_params()
            indexes.add_index(
                field_name="vector",
                index_name="vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
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
        return MilvusSparseSession(
            client=client,
            collection_name=collection_name,
            search_params=dict(self.search_params),
            deployment=self.target.deployment,
        )


class MilvusSparseSession:
    def __init__(
        self,
        *,
        client: Any,
        collection_name: str,
        search_params: Mapping[str, Any],
        deployment: str,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.search_params = dict(search_params)
        self.deployment = deployment
        self._ids: set[str] = set()
        self._sealed = False
        self._closed = False

    def add(self, ids: Sequence[str], vectors: Sequence[SparseVector]) -> None:
        if self._sealed or self._closed:
            raise RuntimeError("Cannot add vectors to a sealed or closed sparse index")
        normalized_ids = [str(item_id) for item_id in ids]
        if len(normalized_ids) != len(vectors):
            raise ValueError("ids and vectors must contain the same number of rows")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("ids must be unique within a batch")
        duplicates = self._ids.intersection(normalized_ids)
        if duplicates:
            raise ValueError(f"Duplicate ids across batches: {sorted(duplicates)}")
        self.client.insert(
            collection_name=self.collection_name,
            data=[{"id": item_id, "vector": vector} for item_id, vector in zip(normalized_ids, vectors, strict=True)],
        )
        self._ids.update(normalized_ids)

    def seal(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot seal a closed sparse index")
        if self._sealed:
            return
        if not self._ids:
            raise ValueError("Cannot seal an empty sparse index")
        self.client.flush(collection_name=self.collection_name, timeout=120)
        self.client.load_collection(collection_name=self.collection_name, timeout=120)
        self._sealed = True

    def search(
        self,
        query_vectors: Sequence[SparseVector],
        *,
        top_k: int,
    ) -> list[list[SearchHit]]:
        if not self._sealed or self._closed:
            raise RuntimeError("The sparse index must be sealed before search")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        results = self.client.search(
            collection_name=self.collection_name,
            data=list(query_vectors),
            anns_field="vector",
            limit=min(top_k, len(self._ids)),
            search_params={"metric_type": "IP", "params": {}, **self.search_params},
            consistency_level="Strong",
        )
        return [[SearchHit(id=str(hit["id"]), score=float(hit["distance"])) for hit in hits] for hits in results]

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "milvus",
            "deployment": self.deployment,
            "metric": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "count": len(self._ids),
            "pymilvus_version": importlib.metadata.version("pymilvus"),
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
        finally:
            self.client.close()
            self._closed = True
