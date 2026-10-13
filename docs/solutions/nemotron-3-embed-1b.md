# Nemotron 3 Embed 1B

NVIDIA Nemotron 3 Embed 1B embeds the shared 768-token chunks through OpenRouter and retrieves 2,048-dimensional vectors with exact cosine search in Milvus before collapsing hits to whole documents.

This Solution adds a lightweight, long-context NVIDIA retrieval model rather than another size of an existing benchmark family. Embeddings are stored in a durable content-addressed cache.
