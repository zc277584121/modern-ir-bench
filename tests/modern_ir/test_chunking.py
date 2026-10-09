from __future__ import annotations

from modern_ir_bench.retrieval import ChunkingPolicy, TokenTextChunker


class CharacterTokenizer:
    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(index, index + 1) for index in range(len(text))]


def test_token_text_chunker_keeps_title_and_uses_overlap() -> None:
    chunker = TokenTextChunker(
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
