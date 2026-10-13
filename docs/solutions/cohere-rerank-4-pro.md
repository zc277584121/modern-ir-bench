# Cohere Rerank 4 Pro

Cohere Rerank 4 Pro reranks 50 complete-document candidates from the maintained BM25/Qwen3 weighted hybrid through OpenRouter, then returns the ten highest-scoring document IDs.

The first-stage rankings and pairwise rerank scores are persisted separately. Re-running the Solution can therefore reuse candidate evidence and completed Cohere scores without repeating either stage.
