# Perplexity Embed V1 4B

This managed dense Solution uses `perplexity/pplx-embed-v1-4b` through
OpenRouter. Documents and queries use the API's `search_document` and
`search_query` input types, respectively. Documents use the benchmark-wide
768-token chunks, and Milvus performs exact cosine search over native
2560-dimensional vectors.

Embeddings are cached by model, dimension, input type, and exact input content.
