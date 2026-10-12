"""Pinned model revisions and shared local inference settings."""

from __future__ import annotations

from typing import Any

EMBEDDING_GEMMA_2_MODEL = "google/embeddinggemma-2"
EMBEDDING_GEMMA_2_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
WEMM_2B_MODEL = "tencent/WeMM-Embedding-2B"
WEMM_2B_REVISION = "bbd6cd4bf52cfc6716f752a2df80b2706720bd95"


def stable_local_model_kwargs(device: str | None = None) -> dict[str, Any]:
    """Use the model-card-recommended stable precision on the available hardware."""

    import torch

    use_cuda = device != "cpu" and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32
    return {"torch_dtype": dtype}
