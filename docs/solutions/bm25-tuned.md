# BM25 · tuned chunks

This Solution keeps the same language-aware analyzers and shared chunk
boundaries as the default chunked BM25 Solution, then applies fixed scoring
parameters per language pool:

- English: `k1=1.8, b=0.75`
- Chinese: `k1=2.0, b=0.5`

The settings were selected with local parameter sweeps and checked with
five-fold out-of-fold selection before the benchmark run. They are a compact
benchmark-specific operating point, not a claim that these parameters are
universally optimal.

The Solution remains lexical-only: it does not use embeddings, query
expansion, reranking, or score fusion. Chunk hits are collapsed to whole-document
document rankings using the same policy as the default chunked Solution.
