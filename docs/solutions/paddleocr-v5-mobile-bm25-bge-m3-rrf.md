# PP-OCRv5 Mobile + BM25/BGE-M3 RRF

This local hybrid uses one OCR representation and two complementary retrieval
methods.

## Pipeline

1. OCR every PDF page with PP-OCRv5 Mobile and concatenate each document.
2. Retrieve the top 50 documents with language-aware Milvus BM25.
3. Retrieve the top 50 documents with BGE-M3 and exact Milvus cosine search.
4. Fuse ranks with equal-weight reciprocal rank fusion using `k = 60`.
5. Return the first ten unique document identifiers.

RRF is used instead of combining raw scores because BM25 and cosine scores are
not calibrated to the same scale. The Solution performs both retrieval paths at
runtime; it is not a post-hoc leaderboard calculation.
