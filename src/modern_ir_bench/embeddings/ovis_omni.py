"""Ovis Omni multimodal embeddings with last-token pooling."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np

from modern_ir_bench.embeddings.sqlite_cache import SqliteVectorCache


class OvisOmniEmbedding:
    """Encode text queries and visual documents with the Ovis Omni Thinker."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        cache_path: Path | None = None,
        batch_size: int = 1,
        device: str | None = None,
        dimension: int = 2048,
        query_instruction: str = "Retrieve a visual document that answers the given search query.",
        document_instruction: str = "Represent the user's input.",
        max_image_pixels: int = 2_007_040,
    ) -> None:
        if not model or not revision:
            raise ValueError("model and revision must be non-empty")
        if batch_size < 1 or dimension < 1 or max_image_pixels < 1:
            raise ValueError("batch size, dimension, and max image pixels must be positive")
        self.model_id = model
        self.revision = revision
        self.cache = SqliteVectorCache(cache_path)
        self.batch_size = batch_size
        self.device = device
        self._dimension = dimension
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction
        self.max_image_pixels = max_image_pixels
        self._processor: Any | None = None
        self._model: Any | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(inputs, kind="document")

    def encode_queries(self, inputs: Sequence[Any]) -> np.ndarray:
        return self._encode(inputs, kind="query")

    def _encode(
        self,
        inputs: Sequence[Any],
        *,
        kind: Literal["document", "query"],
    ) -> np.ndarray:
        values = list(inputs)
        if any(not isinstance(value, (str, Path)) for value in values):
            raise TypeError("OvisOmniEmbedding accepts strings and image Paths")
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)

        keys = [self._cache_key(value, kind) for value in values]
        cached = self.cache.read(keys)
        missing: list[tuple[str, str | Path]] = []
        seen: set[str] = set()
        for key, value in zip(keys, values, strict=True):
            if key not in cached and key not in seen:
                missing.append((key, value))
                seen.add(key)

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._embed([value for _, value in batch], kind=kind)
            additions = {
                key: vector.tolist()
                for (key, _), vector in zip(batch, vectors, strict=True)
            }
            cached.update(additions)
            self.cache.write(additions)
        return np.asarray([cached[key] for key in keys], dtype=np.float32)

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        import torch
        from transformers import (
            AutoConfig,
            AutoProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        config = AutoConfig.from_pretrained(
            self.model_id,
            revision=self.revision,
            subfolder="model",
        )
        processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
            subfolder="model",
        )
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_id,
            revision=self.revision,
            subfolder="model",
            config=config.thinker_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.lm_head = torch.nn.Identity()
        del model.audio_tower
        model.to(device)
        model.eval()
        self.device = device
        self._processor = processor
        self._model = model
        return processor, model

    def _embed(
        self,
        values: list[str | Path],
        *,
        kind: Literal["document", "query"],
    ) -> np.ndarray:
        import torch
        from qwen_vl_utils import process_vision_info

        processor, model = self._load()
        conversations = [self._conversation(value, kind=kind) for value in values]
        texts = [
            processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conversation in conversations
        ]
        images, videos = process_vision_info(conversations)
        inputs = processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(self.device)
            if value.is_floating_point():
                value = value.to(model.dtype)
            inputs[key] = value

        with torch.inference_mode():
            hidden_states = model(**inputs, use_cache=False).logits
            attention_mask = inputs["attention_mask"]
            last_positions = attention_mask.size(1) - 1 - attention_mask.flip(1).argmax(dim=1)
            vectors = hidden_states[
                torch.arange(hidden_states.size(0), device=hidden_states.device),
                last_positions,
            ]
            vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=-1)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Ovis returned {vectors.shape[1]} dimensions; expected {self.dimension}")
        return vectors.cpu().numpy().astype(np.float32, copy=False)

    def _conversation(
        self,
        value: str | Path,
        *,
        kind: Literal["document", "query"],
    ) -> list[dict[str, Any]]:
        instruction = self.query_instruction if kind == "query" else self.document_instruction
        content: str | list[dict[str, Any]]
        if isinstance(value, str):
            content = value
        else:
            content = [
                {
                    "type": "image",
                    "image": str(value),
                    "max_pixels": self.max_image_pixels,
                }
            ]
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content},
        ]

    def _cache_key(self, value: str | Path, kind: str) -> str:
        digest = sha256()
        instruction = self.query_instruction if kind == "query" else self.document_instruction
        digest.update(
            f"ovis-omni\0{self.model_id}\0{self.revision}\0{self.dimension}\0{kind}\0{instruction}\0".encode()
        )
        if isinstance(value, str):
            digest.update(b"text\0")
            digest.update(value.encode())
        else:
            digest.update(b"image\0")
            with value.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
        return digest.hexdigest()

    def release(self) -> None:
        self._processor = None
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
