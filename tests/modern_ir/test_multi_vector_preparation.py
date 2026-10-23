from __future__ import annotations

import hashlib
import json
import weakref

import numpy as np
import pytest

from modern_ir_bench.embeddings import JinaV4MultiVectorEmbedding, SentenceTransformersMultiVectorEmbedding
from modern_ir_bench.embeddings.cached_multi_vector import CachedMultiVectorEmbedding
from modern_ir_bench.embeddings.model_snapshot import pinned_adapter_snapshot
from modern_ir_bench.retrieval import ChunkedMultiVectorRetrievalSolution, RetrievalResource


class CountingEncoder(CachedMultiVectorEmbedding):
    def __init__(self, root, *, batch_size=2, normalize_float32=False):
        super().__init__(
            model="test/encoder",
            revision="pinned",
            dimension=2,
            batch_size=batch_size,
            device="cpu",
            configuration={"dtype": "float32"},
            cache_dir=root,
            normalize_float32=normalize_float32,
        )
        self.calls = []
        self.released = False

    def _load_model(self):
        raise AssertionError("Unexpected model load")

    def _encode_missing(self, inputs, *, role):
        self.calls.append((role, list(inputs)))
        if "fail" in inputs:
            raise RuntimeError("Encoding failed")
        return [np.asarray([[1.0, 0.0]], dtype=np.float32) for _ in inputs]

    def release(self):
        self.released = True


def make_solution(encoder):
    class NoIndex:
        def open(self, **kwargs):
            raise AssertionError("Pre-encoding opened an index")

    return ChunkedMultiVectorRetrievalSolution(
        id="prepared",
        title="Prepared",
        embedding=encoder,
        index=NoIndex(),
        chunker=lambda value: value.split("|"),
        query_input=lambda value: value["text"],
    )


def test_preparation_caches_both_roles_and_fresh_instance_replays_without_encoding(tmp_path):
    encoder = CountingEncoder(tmp_path)
    solution = make_solution(encoder)
    report = solution.pre_encode(
        [RetrievalResource(id="doc", value="one|two|one")], [{"text": "one"}, {"text": "three"}]
    )
    assert (report["documents"], report["queries"]) == (3, 2)
    assert encoder.calls == [("document", ["one", "two"]), ("query", ["one", "three"])]
    assert encoder.released and encoder.cache_only
    cached_bytes = encoder.cache.path.read_bytes()
    cached_mtime = encoder.cache.path.stat().st_mtime_ns
    fresh = CountingEncoder(tmp_path)
    fresh.cache_only = True
    assert len(fresh.encode_documents(["two", "one", "two"])) == 3
    assert len(fresh.encode_queries(["three", "one"])) == 2
    assert fresh.calls == []
    with pytest.raises(ValueError, match="prepare stage"):
        fresh.encode_documents(["three"])
    assert fresh.calls == []
    assert encoder.cache.path.read_bytes() == cached_bytes
    assert encoder.cache.path.stat().st_mtime_ns == cached_mtime


def test_failed_preparation_keeps_completed_batches_and_releases(tmp_path):
    encoder = CountingEncoder(tmp_path, batch_size=1)
    with pytest.raises(RuntimeError, match="Encoding failed"):
        make_solution(encoder).pre_encode([RetrievalResource(id="doc", value="done|fail")], [])
    assert encoder.released and encoder.cache_only
    assert len(encoder.encode_documents(["done"])) == 1
    with pytest.raises(ValueError, match="Missing"):
        encoder.encode_documents(["fail"])


def test_configuration_addressed_caches_and_content_addressed_pages(tmp_path):
    common = {"model": "test/model", "revision": "one", "dimension": 2, "cache_dir": tmp_path}
    baseline = SentenceTransformersMultiVectorEmbedding(**common, model_kwargs={"torch_dtype": "bfloat16"})
    equivalent = SentenceTransformersMultiVectorEmbedding(
        **common, model_kwargs={"torch_dtype": "bfloat16"}, batch_size=32, device="cuda:3", cache_only=True
    )
    assert equivalent.cache.path == baseline.cache.path
    variants = [
        {"model_kwargs": {"torch_dtype": "float32"}},
        {"model_kwargs": {"torch_dtype": "bfloat16"}, "document_length": 128},
        {"model_kwargs": {"torch_dtype": "bfloat16"}, "encode_kwargs": {"max_pixels": 100}},
        {"model_kwargs": {"torch_dtype": "bfloat16"}, "normalize": False},
        {"model_kwargs": {"torch_dtype": "bfloat16"}, "normalize_float32": True},
        {"model_kwargs": {"torch_dtype": "bfloat16"}, "base_revision": "base"},
    ]
    for changed in variants:
        assert SentenceTransformersMultiVectorEmbedding(**common, **changed).cache.path != baseline.cache.path
    page = tmp_path / "page.jpg"
    page.write_bytes(b"first image")
    first = baseline._cache_key(page, role="document")
    duplicate = tmp_path / "other.jpg"
    duplicate.write_bytes(page.read_bytes())
    assert baseline._cache_key(duplicate, role="document") == first
    page.write_bytes(b"second image")
    assert baseline._cache_key(page, role="document") != first


def test_invalid_encoder_output_is_not_cached(tmp_path, monkeypatch):
    encoder = CountingEncoder(tmp_path)
    for output in ([], [np.zeros((1, 3))], [np.full((1, 2), np.nan)]):
        monkeypatch.setattr(encoder, "_encode_missing", lambda *args, output=output, **kwargs: output)
        with pytest.raises(ValueError):
            encoder.encode_documents(["doc"])
        assert not encoder.cache.path.exists()


def test_existing_explicit_cache_identity_is_preserved(tmp_path):
    cache_path = tmp_path / "existing.sqlite3"
    encoder = SentenceTransformersMultiVectorEmbedding(
        model="vidore/colqwen2.5-v0.2", revision="pinned", dimension=128, cache_path=cache_path, cache_only=True
    )
    key = hashlib.sha256(b"vidore/colqwen2.5-v0.2\0pinned\0query\0saved query").hexdigest()
    vectors = np.ones((2, 128), dtype=np.float32)
    encoder.cache.write({key: vectors})
    np.testing.assert_array_equal(encoder.encode_queries(["saved query"])[0], vectors)
    assert encoder._model is None


def test_float32_normalization_is_persisted_and_zero_vectors_are_rejected(tmp_path, monkeypatch):
    encoder = CountingEncoder(tmp_path, normalize_float32=True)
    monkeypatch.setattr(encoder, "_encode_missing", lambda *args, **kwargs: [np.asarray([[0.3, 0.4]])])
    matrix = encoder.encode_documents(["doc"])[0]
    np.testing.assert_allclose(matrix, [[0.6, 0.8]], atol=1e-7)
    saved = encoder.cache.read([encoder._cache_key("doc", role="document")])
    np.testing.assert_array_equal(next(iter(saved.values())), matrix)
    monkeypatch.setattr(encoder, "_encode_missing", lambda *args, **kwargs: [np.zeros((1, 2))])
    with pytest.raises(ValueError, match="Cannot normalize zero"):
        encoder.encode_documents(["empty"])


def test_adapter_backbone_is_pinned_without_editing_official_snapshots(tmp_path, monkeypatch):
    import huggingface_hub
    import huggingface_hub.constants

    adapter, base = tmp_path / "adapter", tmp_path / "base"
    adapter.mkdir()
    base.mkdir()
    original = json.dumps({"base_model_name_or_path": "org/base", "revision": None})
    (adapter / "adapter_config.json").write_text(original)
    (adapter / "modules.json").write_text("[]")
    calls = []

    def snapshot(model, **kwargs):
        calls.append((model, kwargs))
        return str(base if model == "org/base" else adapter)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    resolved = pinned_adapter_snapshot("org/adapter", "adapter-sha", "base-sha", local_files_only=True)
    assert calls == [
        ("org/adapter", {"revision": "adapter-sha", "local_files_only": True}),
        ("org/base", {"revision": "base-sha", "local_files_only": True}),
    ]
    config = json.loads((resolved / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == str(base)
    assert config["revision"] == "base-sha"
    assert (adapter / "adapter_config.json").read_text() == original
    assert (resolved / "modules.json").resolve() == adapter / "modules.json"


def test_jina_native_roles_and_output_conversion(tmp_path):
    import torch

    calls = []

    class NativeModel:
        def encode_image(self, **kwargs):
            calls.append(("image", kwargs))
            return [torch.ones((3, 128), dtype=torch.bfloat16)]

        def encode_text(self, **kwargs):
            calls.append(("text", kwargs))
            return [torch.ones((2, 128), dtype=torch.bfloat16)]

    encoder = JinaV4MultiVectorEmbedding(
        model="jina/test",
        revision="pinned",
        cache_dir=tmp_path,
        cache_only=False,
        max_pixels=602112,
        max_length=128,
    )
    encoder._model = NativeModel()
    page = tmp_path / "page.jpg"
    page.write_bytes(b"image")
    assert encoder.encode_documents([page])[0].shape == (3, 128)
    assert encoder.encode_queries(["中文 query"])[0].dtype == np.float32
    assert calls[0][1] == {
        "images": [str(page)],
        "max_pixels": 602112,
        "task": "retrieval",
        "return_multivector": True,
        "batch_size": 1,
    }
    assert calls[1][1] == {
        "texts": ["中文 query"],
        "max_length": 128,
        "prompt_name": "query",
        "task": "retrieval",
        "return_multivector": True,
        "batch_size": 1,
    }


def test_jina_loads_native_code_and_adapters_from_one_pinned_snapshot(tmp_path, monkeypatch):
    import huggingface_hub
    import torch
    import transformers

    from modern_ir_bench.embeddings import jina_v4

    calls = []

    class NativeModel:
        def to(self, device):
            calls.append(("device", device))
            return self

        def eval(self):
            return self

    def snapshot(model, **kwargs):
        calls.append(("snapshot", model, kwargs))
        return str(tmp_path / "immutable-snapshot")

    def load(path, **kwargs):
        calls.append(("load", path, kwargs))
        return NativeModel()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load)
    monkeypatch.setattr(jina_v4, "version", jina_v4.ENCODING_DEPENDENCIES.__getitem__)
    encoder = JinaV4MultiVectorEmbedding(
        model="jina/test", revision="fixed-revision", cache_dir=tmp_path, device="cpu", local_files_only=True
    )
    assert encoder._load_model() is encoder._load_model()
    assert calls == [
        ("snapshot", "jina/test", {"revision": "fixed-revision", "local_files_only": True}),
        (
            "load",
            str(tmp_path / "immutable-snapshot"),
            {
                "trust_remote_code": True,
                "local_files_only": True,
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
            },
        ),
        ("device", "cpu"),
    ]


def test_jina_rejects_an_unpinned_encoding_runtime_before_loading_weights(tmp_path, monkeypatch):
    from modern_ir_bench.embeddings import jina_v4

    monkeypatch.setattr(jina_v4, "version", lambda name: "5.19.0")
    encoder = JinaV4MultiVectorEmbedding(model="jina/test", revision="pinned", cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="dependency group"):
        encoder._load_model()
    assert not encoder.cache.path.exists()


def test_release_collects_model_reference_cycles(tmp_path):
    encoder = SentenceTransformersMultiVectorEmbedding(model="test", dimension=2, cache_dir=tmp_path)

    class Model:
        def __init__(self):
            self.cycle = self

    encoder._model = Model()
    reference = weakref.ref(encoder._model)
    encoder.release()
    assert reference() is None
