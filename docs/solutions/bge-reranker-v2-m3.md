# BGE Reranker v2 M3

This self-hosted two-stage family first retrieves 50 document candidates, then scores every query-document pair with the pinned multilingual `BAAI/bge-reranker-v2-m3` cross-encoder. The highest ten scores become the submitted ranking.

Two candidate sources are evaluated without changing the reranker:

- Qwen3 Embedding 8B over shared chunks, collapsed to documents.
- Per-query min-max score fusion of chunked BM25 and Qwen3 Embedding 8B with fixed 25/75 weights.

The reranker reads complete candidate documents with an explicit 4,096-token maximum and float16 inference. Exact scores are cached locally by model revision, query text, document id, and document content hash.
