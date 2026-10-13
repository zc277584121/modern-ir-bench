from modern_ir_bench.embeddings import BgeM3Embedding


class _Model:
    def __init__(self) -> None:
        self.stopped = False

    def stop_self_pool(self) -> None:
        self.stopped = True


def test_bge_m3_release_stops_worker_pool() -> None:
    embedding = BgeM3Embedding(revision="test")
    model = _Model()
    embedding._model = model

    embedding.release()

    assert model.stopped
    assert embedding._model is None
