"""Pinned model revisions and shared local inference settings."""

from __future__ import annotations

from typing import Any

EMBEDDING_GEMMA_2_MODEL = "google/embeddinggemma-2"
EMBEDDING_GEMMA_2_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
COLBERT_V2_MODEL = "colbert-ir/colbertv2.0"
COLBERT_V2_REVISION = "c1e84128e85ef755c096a95bdb06b47793b13acf"
COLPALI_V1_3_MODEL = "vidore/colpali-v1.3"
COLPALI_V1_3_REVISION = "b5c6dd62125326e6f0b540c1f7b36e901cdc0a11"
COLQWEN2_V1_0_MODEL = "vidore/colqwen2-v1.0"
COLQWEN2_V1_0_REVISION = "2b6ac8fb37f46a49e4841e599583d00ae8a20117"
COLQWEN_2_5_MODEL = "vidore/colqwen2.5-v0.2"
COLQWEN_2_5_REVISION = "dcbe8d9cede518bce830488364ba0e40c873645b"
COLSMOL_256M_MODEL = "vidore/colSmol-256M"
COLSMOL_256M_REVISION = "c79b633e17e060cc109b11aa2bf92a1517d3bd6f"
COLSMOL_500M_MODEL = "vidore/colSmol-500M"
COLSMOL_500M_REVISION = "0aaa9726104ce485884c7b8faa8a58a72d5fdbe7"
COLMODERNVBERT_MODEL = "ModernVBERT/colmodernvbert"
COLMODERNVBERT_REVISION = "4c604428a6e1a7e33d0fb34837f0cdecbc22026e"
COLMODERNVBERT_BASE_REVISION = "17604b47f51828a5e904557094552bf23fdd9fca"
JINA_V4_MODEL = "jinaai/jina-embeddings-v4"
JINA_V4_REVISION = "853c867b65b749f3c3c72a06868140d842e04f06"
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
