from __future__ import annotations

from modern_ir_bench.retrieval import CanonicalTextChunker, ChunkingPolicy


class CharacterTokenizer:
    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(index, index + 1) for index in range(len(text))]


def test_canonical_chunker_keeps_title_and_uses_overlap() -> None:
    chunker = CanonicalTextChunker(
        CharacterTokenizer(),
        ChunkingPolicy(
            target_tokens=4,
            min_tokens=3,
            max_tokens=5,
            overlap_tokens=1,
            paragraph_search_tokens=0,
        ),
    )

    assert chunker("Title\n\nabcdefgh") == ["Title\n\nabcd", "Title\n\ndefgh"]
