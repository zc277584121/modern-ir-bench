from pathlib import Path

import numpy as np

from modern_ir_bench.embeddings import OvisOmniEmbedding


def _embedding(tmp_path: Path) -> OvisOmniEmbedding:
    return OvisOmniEmbedding(
        model="organization/model",
        revision="0123456789abcdef",
        cache_path=tmp_path / "vectors.sqlite3",
        dimension=2,
    )


def test_ovis_caches_images_by_content(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"same image")
    second.write_bytes(first.read_bytes())
    embedding = _embedding(tmp_path)
    calls: list[tuple[list[str | Path], str]] = []

    def embed(values, *, kind):
        calls.append((values, kind))
        return np.asarray([[1.0, 0.0] for _ in values], dtype=np.float32)

    monkeypatch.setattr(embedding, "_embed", embed)

    assert embedding.encode_documents([first]).shape == (1, 2)
    assert embedding.encode_documents([second]).shape == (1, 2)
    assert calls == [([first], "document")]


def test_ovis_uses_distinct_query_and_candidate_instructions(tmp_path: Path) -> None:
    embedding = _embedding(tmp_path)
    query = embedding._conversation("find the invoice", kind="query")
    image = embedding._conversation(tmp_path / "page.jpg", kind="document")

    assert query[0]["content"] == "Retrieve a visual document that answers the given search query."
    assert query[1]["content"] == "find the invoice"
    assert image[0]["content"] == "Represent the user's input."
    assert image[1]["content"][0]["type"] == "image"
