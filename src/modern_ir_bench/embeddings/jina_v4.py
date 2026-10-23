"""Native Jina v4 retrieval multi-vectors over text queries and page images."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from modern_ir_bench.embeddings.cached_multi_vector import CachedMultiVectorEmbedding

ENCODING_DEPENDENCIES = {"transformers": "4.57.6", "peft": "0.18.1"}


class JinaV4MultiVectorEmbedding(CachedMultiVectorEmbedding):
    """Use the official retrieval adapter and 128-dimensional multi-vector head."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        cache_dir: Path,
        batch_size: int = 1,
        device: str | None = None,
        dtype: str = "bfloat16",
        max_pixels: int = 602112,
        max_length: int = 32768,
        local_files_only: bool = False,
        cache_only: bool = True,
    ) -> None:
        if dtype not in ("float32", "float16", "bfloat16") or min(max_pixels, max_length) < 1:
            raise ValueError("Invalid dtype or input budget")
        self.dtype = dtype
        self.max_pixels = max_pixels
        self.max_length = max_length
        self.local_files_only = local_files_only
        super().__init__(
            model=model,
            revision=revision,
            dimension=128,
            batch_size=batch_size,
            device=device,
            cache_dir=cache_dir,
            cache_only=cache_only,
            normalize_float32=True,
            configuration={
                "encoding_dependencies": ENCODING_DEPENDENCIES,
                "task": "retrieval",
                "prompt_name": "query",
                "return_multivector": True,
                "dtype": dtype,
                "attention": "sdpa",
                "max_pixels": max_pixels,
                "max_length": max_length,
            },
        )

    def _load_model(self) -> Any:
        if self._model is None:
            if any(version(package) != expected for package, expected in ENCODING_DEPENDENCIES.items()):
                raise RuntimeError(
                    "Jina native encoding requires the pinned 'jina' dependency group in an isolated environment; "
                    "use UV_PROJECT_ENVIRONMENT=.runtime/venvs/jina uv sync --group jina --group multimodal --group pdf"
                )
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModel

            # A local immutable snapshot also pins the native loader's nested adapters/* load.
            snapshot = snapshot_download(self.model_id, revision=self.revision, local_files_only=self.local_files_only)
            self._model = (
                AutoModel.from_pretrained(
                    snapshot,
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=getattr(torch, self.dtype),
                    attn_implementation="sdpa",
                )
                .to(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
                .eval()
            )
        return self._model

    def _encode_missing(self, inputs: Sequence[Any], *, role: str) -> Sequence[Any]:
        model = self._load_model()
        kwargs = {"task": "retrieval", "return_multivector": True, "batch_size": self.batch_size}
        if role == "document":
            vectors = model.encode_image(
                images=[str(value) if isinstance(value, Path) else value for value in inputs],
                max_pixels=self.max_pixels,
                **kwargs,
            )
        else:
            vectors = model.encode_text(texts=list(inputs), prompt_name="query", max_length=self.max_length, **kwargs)
        return [vector.detach().float().cpu().numpy() for vector in vectors]
