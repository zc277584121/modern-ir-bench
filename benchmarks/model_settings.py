"""Pinned model revisions and shared local inference settings."""

from __future__ import annotations

from typing import Any

EMBEDDING_GEMMA_2_MODEL = "google/embeddinggemma-2"
EMBEDDING_GEMMA_2_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
COLBERT_V2_MODEL = "colbert-ir/colbertv2.0"
COLBERT_V2_REVISION = "c1e84128e85ef755c096a95bdb06b47793b13acf"
JINA_COLBERT_V2_MODEL = "jinaai/jina-colbert-v2"
JINA_COLBERT_V2_REVISION = "a9dc5cd7293d4c71dbbba04829923ba4d0e4f6ea"
JINA_XLM_ROBERTA_CODE_REVISION = "bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3"
MLATEON_MODEL = "lightonai/mLateOn"
MLATEON_REVISION = "bbd883ae1fadb563056ff73b1061ae8172a3fcda"
WEMM_2B_MODEL = "tencent/WeMM-Embedding-2B"
WEMM_2B_REVISION = "bbd6cd4bf52cfc6716f752a2df80b2706720bd95"
OVIS_OMNI_3B_MODEL = "ATH-MaaS/Ovis-Omni-Embedding-3B"
OVIS_OMNI_3B_REVISION = "9726d1ad56420a07ba745aafaf5d7030963cda4e"


def stable_local_model_kwargs(device: str | None = None) -> dict[str, Any]:
    """Use the model-card-recommended stable precision on the available hardware."""

    import torch

    use_cuda = device != "cpu" and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32
    return {"torch_dtype": dtype}
