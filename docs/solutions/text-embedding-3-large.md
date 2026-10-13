# OpenAI Text Embedding 3 Large · text

This managed dense Solution uses `openai/text-embedding-3-large` through
OpenRouter. Documents and queries use the API's `search_document` and
`search_query` input types, respectively. Documents use the benchmark-wide
768-token chunks, and Milvus performs exact cosine search over native
3072-dimensional vectors.

Embeddings are cached locally by model, dimension, input type, and exact input
content. The cache is runtime state and is not part of the repository or the
published result artifact.
