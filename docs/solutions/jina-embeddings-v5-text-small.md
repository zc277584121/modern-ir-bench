# Jina Embeddings v5 Text Small

Jina Embeddings v5 Text Small embeds the shared 768-token chunks through the Jina API, distinguishes retrieval passages from retrieval queries, and retrieves chunk vectors with exact cosine search in Milvus before collapsing hits to whole documents.

The run uses 1,024-dimensional normalized vectors and a durable content-addressed embedding cache. It represents Jina's current production-oriented multilingual text embedding family without adding another size variant of the same model.
