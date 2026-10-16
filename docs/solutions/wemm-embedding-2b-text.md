# WeMM Embedding 2B · text retrieval

This self-hosted dense Solution pins `tencent/WeMM-Embedding-2B` to revision
`bbd6cd4bf52cfc6716f752a2df80b2706720bd95`. It uses the model's dedicated
`encode_document` and `encode_query` routes to embed the benchmark's shared
768-token text chunks as 2,048-dimensional, L2-normalized vectors.

Milvus performs exact cosine search with a FLAT index. Chunk hits are collapsed
to whole-document rankings without reranking. The chunk boundaries and retrieval
settings match the other chunked dense Solutions so the comparison isolates the
embedding model.
