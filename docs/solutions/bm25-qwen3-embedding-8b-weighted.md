# BM25 + Qwen3 Embedding 8B · Weighted score fusion

This hybrid family retrieves the Top 50 documents independently with Milvus
BM25 and Qwen3 Embedding 8B over the shared 768-token chunks. Within each query,
the two score lists are independently min-max normalized and then combined with
fixed BM25/dense weights. Milvus exposes its BM25 relevance as a distance where
lower values rank first, while cosine similarity ranks higher values first; the
normalization preserves those respective directions.

The published 25/75, 50/50, and 75/25 variants are predefined ablations. They
share the same component candidates and do not select a weight by optimizing on
the benchmark labels.
