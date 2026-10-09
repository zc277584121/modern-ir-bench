"""Model-independent token chunk boundaries for retrieval Solutions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

CANONICAL_TOKENIZER = "BAAI/bge-m3"
CANONICAL_TOKENIZER_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class OffsetTokenizer(Protocol):
    def offsets(self, text: str) -> Sequence[tuple[int, int]]: ...


class HuggingFaceOffsetTokenizer:
    """Use one pinned tokenizer only to define shared chunk boundaries."""

    def __init__(
        self,
        *,
        model: str = CANONICAL_TOKENIZER,
        revision: str = CANONICAL_TOKENIZER_REVISION,
    ) -> None:
        from transformers import AutoTokenizer

        self.model = model
        self.revision = revision
        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            use_fast=True,
        )

    def offsets(self, text: str) -> Sequence[tuple[int, int]]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return [tuple(item) for item in encoded["offset_mapping"]]


@dataclass(frozen=True, slots=True)
class ChunkingPolicy:
    target_tokens: int = 768
    min_tokens: int = 384
    max_tokens: int = 896
    overlap_tokens: int = 128
    paragraph_search_tokens: int = 128

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_tokens < self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("expected overlap < minimum <= target <= maximum")


class CanonicalTextChunker:
    """Chunk a title/body resource once, independently of the embedding model."""

    def __init__(
        self,
        tokenizer: OffsetTokenizer,
        policy: ChunkingPolicy | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.policy = policy or ChunkingPolicy()

    def __call__(self, content: str) -> list[str]:
        title, separator, body = content.partition("\n\n")
        if not separator:
            title = ""
            body = content
        offsets = list(self.tokenizer.offsets(body))
        if not offsets:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(offsets):
            end = self._choose_end(body, offsets, start)
            start_char = offsets[start][0]
            end_char = offsets[end - 1][1]
            chunk = body[start_char:end_char]
            chunks.append(f"{title}\n\n{chunk}" if title else chunk)
            if end == len(offsets):
                break
            start = max(start + 1, end - self.policy.overlap_tokens)
        return chunks

    def _choose_end(
        self,
        text: str,
        offsets: Sequence[tuple[int, int]],
        start: int,
    ) -> int:
        remaining = len(offsets) - start
        if remaining <= self.policy.max_tokens:
            return len(offsets)

        lower = start + self.policy.min_tokens
        target = start + self.policy.target_tokens
        upper = min(len(offsets), start + self.policy.max_tokens)
        radius = self.policy.paragraph_search_tokens
        candidates: list[int] = []
        for end in range(max(lower, target - radius), min(upper, target + radius) + 1):
            char_end = offsets[end - 1][1]
            if text[char_end : char_end + 2] == "\n\n" or text[max(0, char_end - 2) : char_end] == "\n\n":
                candidates.append(end)
        return min(candidates, key=lambda item: abs(item - target)) if candidates else min(target, upper)
