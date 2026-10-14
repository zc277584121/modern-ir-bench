"""Milvus BM25 Solutions for full-document and chunked retrieval."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.retrieval._chunk_ranking import (
    candidate_depths,
    collapse_chunk_rankings,
    complete_document_rankings,
    has_enough_documents,
)
from modern_ir_bench.retrieval.indexes.milvus.index import _collection_name
from modern_ir_bench.retrieval.indexes.milvus.target import MilvusTarget
from modern_ir_bench.retrieval.types import RetrievalResource, SearchHit, TextInput


@dataclass(frozen=True, kw_only=True)
class BM25LanguageConfig:
    """Analyzer and scoring parameters for one language pool."""

    analyzer: Mapping[str, Any]
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")


def default_bm25_language_configs() -> Mapping[str, BM25LanguageConfig]:
    return {
        "en": BM25LanguageConfig(analyzer={"tokenizer": "standard", "filter": ["lowercase"]}),
        "zh": BM25LanguageConfig(analyzer={"tokenizer": "jieba"}),
    }


@dataclass(frozen=True, kw_only=True)
class MilvusBM25Solution(Solution):
    target: MilvusTarget
    chunker: Callable[[str], Sequence[str]] | None = None
    language_configs: Mapping[str, BM25LanguageConfig] = field(default_factory=default_bm25_language_configs)
    candidate_multiplier: int = 3
    insert_batch_size: int = 256
    collection_prefix: str = "modern_ir_bm25"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.candidate_multiplier < 1 or self.insert_batch_size < 1:
            raise ValueError("candidate multiplier and insert batch size must be positive")

    def prepare(
        self,
        resources: Iterable[RetrievalResource[TextInput]],
    ) -> MilvusBM25Session:
        rows = list(resources)
        if not rows:
            raise ValueError("A BM25 Solution requires at least one resource")
        languages = {resource.value.language for resource in rows}
        if None in languages or len(languages) != 1:
            raise ValueError("BM25 resources must belong to one declared language pool")
        language = str(next(iter(languages)))
        if language not in self.language_configs:
            raise ValueError(f"No BM25 configuration declared for language: {language}")
        language_config = self.language_configs[language]

        from pymilvus import DataType, Function, FunctionType, MilvusClient

        arguments = dict(self.target.client_arguments())
        if self.target.deployment == "milvus-lite":
            Path(arguments["uri"]).parent.mkdir(parents=True, exist_ok=True)
        client = MilvusClient(**arguments)
        collection_name = _collection_name(self.collection_prefix)
        try:
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, max_length=2048, is_primary=True)
            schema.add_field("document_id", DataType.VARCHAR, max_length=2048)
            schema.add_field(
                "content",
                DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params=dict(language_config.analyzer),
            )
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(
                Function(
                    name="bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["content"],
                    output_field_names=["sparse"],
                )
            )
            indexes = MilvusClient.prepare_index_params()
            indexes.add_index(
                field_name="sparse",
                index_name="bm25_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    "bm25_k1": language_config.k1,
                    "bm25_b": language_config.b,
                },
            )
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=indexes,
                consistency_level="Strong",
            )

            units: list[dict[str, str]] = []
            unit_to_document: dict[str, str] = {}
            for resource in rows:
                texts = list(self.chunker(resource.value.text)) if self.chunker is not None else [resource.value.text]
                for chunk_index, text in enumerate(texts):
                    unit_id = f"{resource.id}:chunk:{chunk_index:04d}" if self.chunker is not None else resource.id
                    units.append({"id": unit_id, "document_id": resource.id, "content": text})
                    unit_to_document[unit_id] = resource.id
            for start in range(0, len(units), self.insert_batch_size):
                client.insert(
                    collection_name=collection_name,
                    data=units[start : start + self.insert_batch_size],
                )
            client.flush(collection_name=collection_name, timeout=120)
            client.load_collection(collection_name=collection_name, timeout=120)
        except Exception:
            if client.has_collection(collection_name):
                client.drop_collection(collection_name)
            client.close()
            raise
        return MilvusBM25Session(
            client=client,
            collection_name=collection_name,
            language=language,
            language_config=language_config,
            unit_to_document=unit_to_document,
            document_count=len(rows),
            candidate_multiplier=self.candidate_multiplier,
        )


class MilvusBM25Session:
    def __init__(
        self,
        *,
        client: Any,
        collection_name: str,
        language: str,
        language_config: BM25LanguageConfig,
        unit_to_document: Mapping[str, str],
        document_count: int,
        candidate_multiplier: int,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.language = language
        self.language_config = language_config
        self.unit_to_document = dict(unit_to_document)
        self.document_count = document_count
        self.candidate_multiplier = candidate_multiplier

    def search_batch(
        self,
        queries: Sequence[TextInput],
        *,
        top_k: int,
    ) -> list[list[SearchHit]]:
        if any(query.language != self.language for query in queries):
            raise ValueError("Query language does not match the prepared BM25 pool")
        for candidate_depth in candidate_depths(
            unit_count=len(self.unit_to_document),
            top_k=top_k,
            initial_multiplier=self.candidate_multiplier,
        ):
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query.text for query in queries],
                anns_field="sparse",
                limit=candidate_depth,
                search_params={"metric_type": "BM25", "params": {}},
                consistency_level="Strong",
            )
            unit_rankings = [
                [SearchHit(id=str(hit["id"]), score=float(hit["distance"])) for hit in unit_hits]
                for unit_hits in results
            ]
            document_rankings = collapse_chunk_rankings(
                unit_rankings,
                unit_to_document=self.unit_to_document,
                top_k=top_k,
            )
            if has_enough_documents(
                document_rankings,
                document_count=self.document_count,
                top_k=top_k,
            ):
                return document_rankings
            if candidate_depth == len(self.unit_to_document):
                return complete_document_rankings(
                    document_rankings,
                    document_ids=self.unit_to_document.values(),
                    top_k=top_k,
                )
        raise AssertionError("candidate depth iteration must return")

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "milvus-bm25",
            "language": self.language,
            "analyzer": dict(self.language_config.analyzer),
            "k1": self.language_config.k1,
            "b": self.language_config.b,
            "units": len(self.unit_to_document),
            "documents": self.document_count,
            "candidate_multiplier": self.candidate_multiplier,
        }

    def close(self) -> None:
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
        finally:
            self.client.close()
