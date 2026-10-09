from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

from modern_ir_bench.embeddings import SentenceTransformersEmbedding


def test_document_and_query_prompts_are_independent(monkeypatch) -> None:
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((len(texts), 2), dtype=np.float32)

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embedding = SentenceTransformersEmbedding(
        model="fake",
        document_prompt="Document: ",
        query_prompt_name="query",
    )
    embedding.encode_documents(["document"])
    embedding.encode_queries(["query"])

    assert calls[0][1]["prompt"] == "Document: "
    assert "prompt_name" not in calls[0][1]
    assert calls[1][1]["prompt_name"] == "query"
    assert "prompt" not in calls[1][1]
