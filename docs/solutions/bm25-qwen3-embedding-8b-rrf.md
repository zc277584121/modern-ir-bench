# BM25 + Qwen3 Embedding 8B · RRF

This hybrid Solution prepares the shared 768-token chunks twice: Milvus BM25
provides lexical ranks and Qwen3 Embedding 8B provides dense semantic ranks.
For each query it retrieves the Top 50 documents from both components and fuses
them with equal-weight reciprocal rank fusion using a rank constant of 60.

RRF uses ranks rather than raw scores, so the lexical and cosine score scales do
not need calibration.
