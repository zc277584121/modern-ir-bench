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


def test_model_configuration_and_encode_routes_are_forwarded(monkeypatch) -> None:
    constructors = []
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            constructors.append((args, kwargs))

        def encode_document(self, values, **kwargs):
            calls.append(("document", values, kwargs))
            return np.ones((len(values), 3), dtype=np.float32)

        def encode_query(self, values, **kwargs):
            calls.append(("query", values, kwargs))
            return np.ones((len(values), 3), dtype=np.float32)

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embedding = SentenceTransformersEmbedding(
        model="fake",
        dimension=3,
        model_kwargs={"torch_dtype": "auto"},
        config_kwargs={"audio_config": None},
        encode_kwargs={"truncate_dim": 3},
        document_method="encode_document",
        query_method="encode_query",
    )
    embedding.encode_documents([{"image": "page.jpg"}])
    embedding.encode_queries(["query"])

    assert constructors[0][1]["model_kwargs"] == {"torch_dtype": "auto"}
    assert constructors[0][1]["config_kwargs"] == {"audio_config": None}
    assert calls[0][0] == "document"
    assert calls[0][2]["truncate_dim"] == 3
    assert calls[1][0] == "query"


def test_multi_process_pool_is_reused_and_released(monkeypatch) -> None:
    calls = []
    pool = {"processes": []}

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start_multi_process_pool(self, *, target_devices):
            calls.append(("start", target_devices))
            return pool

        def encode_document(self, values, **kwargs):
            calls.append(("encode", kwargs["pool"]))
            return np.ones((len(values), 3), dtype=np.float32)

        def stop_multi_process_pool(self, value) -> None:
            calls.append(("stop", value))

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embedding = SentenceTransformersEmbedding(
        model="fake",
        dimension=3,
        encode_devices=("cuda:0", "cuda:1"),
        document_method="encode_document",
    )
    embedding.encode_documents(["first"])
    embedding.encode_documents(["second"])
    embedding.release()

    assert calls == [
        ("start", ["cuda:0", "cuda:1"]),
        ("encode", pool),
        ("encode", pool),
        ("stop", pool),
    ]
