# DashScope Text Embedding v4

DashScope Text Embedding v4 embeds the shared 768-token chunks through Alibaba Cloud Model Studio's native API and retrieves them with exact cosine search in Milvus before collapsing hits to whole documents.

The native API preserves the model's asymmetric `document` and `query` modes. Query embeddings also receive the benchmark's broad web-retrieval instruction. Vectors use the recommended 1,024-dimensional setting and a durable content-addressed cache.
