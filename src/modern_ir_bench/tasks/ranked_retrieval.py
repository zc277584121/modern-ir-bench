"""Ranked retrieval over a task-owned three-table dataset contract."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from datasets import Dataset, Features, Value

from modern_ir_bench.core.solution import Solution
from modern_ir_bench.core.task import Task
from modern_ir_bench.retrieval import MappedResourceSource


class RankedRetrieval(Task):
    """Retrieve relevant documents from one independent candidate pool."""

    dataset_features: ClassVar[dict[str, Features]] = {
        "documents": Features(
            {
                "document_id": Value("string"),
                "content": Value("string"),
            }
        ),
        "queries": Features(
            {
                "query_id": Value("string"),
                "query": Value("string"),
            }
        ),
        "qrels": Features(
            {
                "query_id": Value("string"),
                "document_id": Value("string"),
                "relevance": Value("int32"),
            }
        ),
    }

    def __init__(
        self,
        *,
        top_k: int = 100,
        query_batch_size: int = 32,
        document_value: Callable[[Mapping[str, Any]], Any] | None = None,
        query_value: Callable[[Mapping[str, Any]], Any] | None = None,
        pool_field: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")
        self.top_k = top_k
        self.query_batch_size = query_batch_size
        self.document_value = document_value or (lambda row: row["content"])
        self.query_value = query_value or (lambda row: row["query"])
        self.pool_field = pool_field

    def validate_dataset(self, dataset: Any) -> None:
        if set(dataset) != set(self.dataset_features):
            raise ValueError("Ranked retrieval data requires documents, queries, and qrels tables")
        for table_name, expected_features in self.dataset_features.items():
            table = dataset[table_name]
            missing = set(expected_features).difference(table.features)
            mismatched = {
                field
                for field in expected_features
                if field in table.features and table.features[field] != expected_features[field]
            }
            if missing or mismatched:
                raise ValueError(
                    f"Unexpected features for {table_name}: missing={sorted(missing)}, mismatched={sorted(mismatched)}"
                )

        document_ids = dataset["documents"]["document_id"]
        query_ids = dataset["queries"]["query_id"]
        if not document_ids or len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be non-empty and unique")
        if not query_ids or len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be non-empty and unique")
        if any(not value for value in dataset["documents"]["content"]):
            raise ValueError("document content must be non-empty")
        if any(not value for value in dataset["queries"]["query"]):
            raise ValueError("query text must be non-empty")

        document_id_set = set(document_ids)
        query_id_set = set(query_ids)
        qrel_pairs: set[tuple[str, str]] = set()
        covered_queries: set[str] = set()
        for row in dataset["qrels"]:
            pair = (row["query_id"], row["document_id"])
            if pair in qrel_pairs:
                raise ValueError(f"Duplicate qrel: {pair}")
            if row["query_id"] not in query_id_set or row["document_id"] not in document_id_set:
                raise ValueError(f"Dangling qrel: {pair}")
            if row["relevance"] < 0:
                raise ValueError(f"Qrel relevance must be non-negative: {pair}")
            qrel_pairs.add(pair)
            if row["relevance"] > 0:
                covered_queries.add(row["query_id"])
        if covered_queries != query_id_set:
            raise ValueError("Every query must have at least one positive qrel")
        if self.pool_field is not None:
            for table_name in ("documents", "queries"):
                if self.pool_field not in dataset[table_name].column_names:
                    raise ValueError(f"Pool field {self.pool_field!r} is missing from {table_name}")
            document_pool = dict(
                zip(
                    dataset["documents"]["document_id"],
                    dataset["documents"][self.pool_field],
                    strict=True,
                )
            )
            query_pool = dict(
                zip(
                    dataset["queries"]["query_id"],
                    dataset["queries"][self.pool_field],
                    strict=True,
                )
            )
            for row in dataset["qrels"]:
                if document_pool[row["document_id"]] != query_pool[row["query_id"]]:
                    raise ValueError(f"Qrel crosses independent pools: {row['query_id']} -> {row['document_id']}")

    def evaluate(self, *, dataset: Any, solution: Solution) -> Dataset:
        if self.pool_field is None:
            return Dataset.from_list(
                self._evaluate_rows(
                    documents=dataset["documents"],
                    queries=dataset["queries"],
                    qrels=dataset["qrels"],
                    solution=solution,
                )
            )

        observations: list[dict[str, Any]] = []
        for pool in sorted(set(dataset["queries"][self.pool_field])):
            documents = dataset["documents"].filter(
                lambda row, selected_pool=pool: row[self.pool_field] == selected_pool,
                desc=f"Select {pool} document pool",
            )
            queries = dataset["queries"].filter(
                lambda row, selected_pool=pool: row[self.pool_field] == selected_pool,
                desc=f"Select {pool} query pool",
            )
            query_ids = set(queries["query_id"])
            qrels = dataset["qrels"].filter(
                lambda row, selected_query_ids=query_ids: row["query_id"] in selected_query_ids,
                desc=f"Select {pool} Qrels",
            )
            observations.extend(
                self._evaluate_rows(
                    documents=documents,
                    queries=queries,
                    qrels=qrels,
                    solution=solution,
                )
            )
        return Dataset.from_list(observations)

    def _evaluate_rows(
        self,
        *,
        documents: Dataset,
        queries: Dataset,
        qrels: Dataset,
        solution: Solution,
    ) -> list[dict[str, Any]]:
        expected: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for row in qrels:
            expected[row["query_id"]].append((row["document_id"], row["relevance"]))

        resources = MappedResourceSource(
            documents,
            id_of=lambda row: row["document_id"],
            value_of=self.document_value,
        )
        document_ids = set(documents["document_id"])
        searcher = solution.prepare(resources)
        observations = []
        try:
            for batch in queries.iter(batch_size=self.query_batch_size):
                rows = [{column: batch[column][index] for column in batch} for index in range(len(batch["query_id"]))]
                results = searcher.search_batch(
                    [self.query_value(row) for row in rows],
                    top_k=self.top_k,
                )
                if len(results) != len(batch["query_id"]):
                    raise RuntimeError("Solution returned a different number of rankings than queries")
                for query_id, hits in zip(batch["query_id"], results, strict=True):
                    ranked_ids = [hit.id for hit in hits]
                    if len(ranked_ids) != len(set(ranked_ids)):
                        raise RuntimeError(f"Solution returned duplicate hits for query {query_id}")
                    if not set(ranked_ids).issubset(document_ids):
                        raise RuntimeError(f"Solution returned unknown hits for query {query_id}")
                    relevant_documents = [
                        (document_id, relevance) for document_id, relevance in expected[query_id] if relevance > 0
                    ]
                    observations.append(
                        {
                            "query_id": query_id,
                            "expected_ids": [document_id for document_id, _ in relevant_documents],
                            "expected_relevance": [relevance for _, relevance in relevant_documents],
                            "ranked_ids": ranked_ids,
                            "scores": [hit.score for hit in hits],
                        }
                    )
        finally:
            searcher.close()
        return observations

    def observations_from_rankings(
        self,
        dataset: Any,
        rankings: Mapping[str, Sequence[str]],
        *,
        validate: bool = True,
    ) -> Dataset:
        """Create metric inputs from an immutable query-id-to-document ranking artifact."""

        if validate:
            self.validate_dataset(dataset)
        query_ids = set(dataset["queries"]["query_id"])
        if set(rankings) != query_ids:
            raise ValueError("Published rankings must cover every query exactly once")
        document_ids = set(dataset["documents"]["document_id"])
        document_pool = None
        query_pool = None
        if self.pool_field is not None:
            document_pool = dict(
                zip(
                    dataset["documents"]["document_id"],
                    dataset["documents"][self.pool_field],
                    strict=True,
                )
            )
            query_pool = dict(
                zip(
                    dataset["queries"]["query_id"],
                    dataset["queries"][self.pool_field],
                    strict=True,
                )
            )
        expected: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for row in dataset["qrels"]:
            if row["relevance"] > 0:
                expected[row["query_id"]].append((row["document_id"], row["relevance"]))
        observations = []
        for query_id in dataset["queries"]["query_id"]:
            ranked_ids = list(rankings[query_id])
            if len(ranked_ids) != len(set(ranked_ids)):
                raise ValueError(f"Published ranking contains duplicates for {query_id}")
            if not set(ranked_ids).issubset(document_ids):
                raise ValueError(f"Published ranking contains unknown documents for {query_id}")
            if document_pool is not None and query_pool is not None:
                wrong_pool = [
                    document_id for document_id in ranked_ids if document_pool[document_id] != query_pool[query_id]
                ]
                if wrong_pool:
                    raise ValueError(f"Published ranking crosses independent pools for {query_id}: {wrong_pool}")
            qrels = expected[query_id]
            observations.append(
                {
                    "query_id": query_id,
                    "expected_ids": [document_id for document_id, _ in qrels],
                    "expected_relevance": [relevance for _, relevance in qrels],
                    "ranked_ids": ranked_ids,
                }
            )
        return Dataset.from_list(observations)
