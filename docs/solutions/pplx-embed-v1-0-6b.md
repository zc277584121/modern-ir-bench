# Perplexity Embed V1 0.6B

Perplexity Embed V1 0.6B embeds the shared 768-token chunks through OpenRouter and retrieves 1,024-dimensional vectors with exact cosine search in Milvus before collapsing hits to whole documents.

It forms a controlled small-versus-large comparison with Perplexity Embed V1 4B under the same retrieval pipeline. Embeddings are stored in a durable content-addressed cache.
