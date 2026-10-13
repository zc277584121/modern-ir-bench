# OpenAI Text Embedding 3 Small

OpenAI Text Embedding 3 Small embeds the shared 768-token chunks through OpenRouter and retrieves 1,536-dimensional vectors with exact cosine search in Milvus before collapsing hits to whole documents.

It forms a practical cost and model-size comparison with Text Embedding 3 Large under an otherwise identical retrieval pipeline. Embeddings are stored in a durable content-addressed cache.
