"""Cached Voyage multimodal embeddings for text and page images."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from time import sleep
from typing import Any, Literal

import numpy as np

from modern_ir_bench.embeddings.voyage import VoyageEmbedding


class VoyageMultimodalEmbedding(VoyageEmbedding):
    """Encode text queries and rendered page images into one vector space."""

    def __init__(
        self,
        *,
        model: str = "voyage-multimodal-3.5",
        dimension: int = 1024,
        batch_size: int = 16,
        cache_path: Path | None = None,
        max_rate_limit_retries: int = 5,
    ) -> None:
        super().__init__(
            model=model,
            dimension=dimension,
            batch_size=batch_size,
            cache_path=cache_path,
            max_rate_limit_retries=max_rate_limit_retries,
        )

    def _encode(
        self,
        inputs: Sequence[Any],
        *,
        kind: Literal["document", "query"],
    ) -> np.ndarray:
        values = list(inputs)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        keys = [self._multimodal_cache_key(value, kind) for value in values]
        cached = self._read_cache(keys)
        missing = []
        seen: set[str] = set()
        for key, value in zip(keys, values, strict=True):
            if key not in cached and key not in seen:
                missing.append((key, value))
                seen.add(key)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            response = self._multimodal_embed_with_retry([value for _, value in batch], kind)
            embedded = {
                key: self._validate_vector(vector)
                for (key, _), vector in zip(batch, response.embeddings, strict=True)
            }
            cached.update(embedded)
            self._write_cache(embedded)
        return np.asarray([cached[key] for key in keys], dtype=np.float32)

    def _multimodal_embed_with_retry(self, values: list[Any], kind: str) -> Any:
        try:
            from voyageai.error import RateLimitError

            rate_limit_errors: tuple[type[BaseException], ...] = (RateLimitError,)
        except ImportError:
            rate_limit_errors = ()

        for attempt in range(self.max_rate_limit_retries + 1):
            try:
                with ExitStack() as stack:
                    inputs = [[self._api_value(value, stack)] for value in values]
                    return self._get_client().multimodal_embed(
                        inputs,
                        model=self.model_id,
                        input_type=kind,
                        output_dimension=self.dimension,
                    )
            except rate_limit_errors:
                if attempt == self.max_rate_limit_retries:
                    raise
                sleep(min(120, 10 * 2**attempt))
        raise AssertionError("unreachable")

    @staticmethod
    def _api_value(value: Any, stack: ExitStack) -> Any:
        if isinstance(value, Path):
            from PIL import Image

            return stack.enter_context(Image.open(value))
        return value

    def _multimodal_cache_key(self, value: Any, kind: str) -> str:
        digest = sha256()
        digest.update(f"{self.model_id}\0{self.dimension}\0{kind}\0".encode())
        if isinstance(value, Path):
            with value.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
        elif isinstance(value, str):
            digest.update(value.encode())
        else:
            from PIL import Image

            if not isinstance(value, Image.Image):
                raise TypeError(f"Unsupported multimodal input: {type(value).__name__}")
            digest.update(value.mode.encode())
            digest.update(str(value.size).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()
