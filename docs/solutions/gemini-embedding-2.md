# Gemini Embedding 2 · text

This managed dense Solution uses `google/gemini-embedding-2` through OpenRouter.
Documents and queries use the API's `search_document` and `search_query` input
types, respectively. Documents use the benchmark-wide 768-token chunks, and
Milvus performs exact cosine search over native 3072-dimensional vectors.

Embeddings are cached by model, dimension, input type, and exact input content.
