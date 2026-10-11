from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from modern_ir_bench.embeddings import VoyageMultimodalEmbedding


class FakeVoyageClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def multimodal_embed(self, inputs, **kwargs):
        self.calls.append({"inputs": inputs, **kwargs})
        dimension = int(kwargs["output_dimension"])
        return SimpleNamespace(embeddings=[[float(index + 1)] * dimension for index in range(len(inputs))])


def test_multimodal_embedding_caches_images_and_text(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (16, 24), "white").save(image_path)
    client = FakeVoyageClient()
    embedding = VoyageMultimodalEmbedding(
        dimension=4,
        batch_size=2,
        cache_path=tmp_path / "cache.sqlite3",
    )
    embedding._client = client

    first_documents = embedding.encode_documents([image_path, image_path])
    second_documents = embedding.encode_documents([image_path])
    first_queries = embedding.encode_queries(["find the white page"])
    second_queries = embedding.encode_queries(["find the white page"])

    assert first_documents.shape == (2, 4)
    assert second_documents.shape == (1, 4)
    assert first_queries.shape == (1, 4)
    assert second_queries.shape == (1, 4)
    assert len(client.calls) == 2
    assert client.calls[0]["input_type"] == "document"
    assert client.calls[1]["input_type"] == "query"
