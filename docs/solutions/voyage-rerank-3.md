# Voyage Rerank 3

This two-stage family first retrieves 50 document candidates, then sends each query and the complete candidate documents to Voyage `rerank-3`. The returned relevance scores replace the first-stage order; the best ten documents become the submitted ranking.

Two candidate sources are evaluated without changing the reranker:

- Qwen3 Embedding 8B over shared chunks, collapsed to documents.
- Per-query min-max score fusion of chunked BM25 and Qwen3 Embedding 8B with fixed 25/75 weights.

The reranker receives one query and all 50 candidate documents in one request. Voyage truncation is enabled only when a single query-document pair exceeds the model limit. Exact query-document scores are cached locally by model, query text, document id, and document content hash.
