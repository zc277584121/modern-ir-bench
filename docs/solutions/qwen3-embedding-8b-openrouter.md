# Qwen3 Embedding 8B via OpenRouter

This managed dense Solution calls `qwen/qwen3-embedding-8b` through OpenRouter's
Embeddings API. Documents and queries use the API's `search_document` and
`search_query` input types respectively. Documents use the shared benchmark
chunks, and Milvus performs exact cosine search over 4096-dimensional vectors.

Responses are cached by model, dimension, input type, and exact input text so an
interrupted run can resume without purchasing the same embedding twice.
