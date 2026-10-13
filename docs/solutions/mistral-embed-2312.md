# Mistral Embed 2312

Mistral Embed 2312 embeds the shared 768-token chunks through OpenRouter and retrieves 1,024-dimensional vectors with exact cosine search in Milvus before collapsing hits to whole documents.

It is retained as the current OpenRouter-accessible Mistral text-retrieval baseline and provides a European commercial-provider comparison. Embeddings are stored in a durable content-addressed cache.
