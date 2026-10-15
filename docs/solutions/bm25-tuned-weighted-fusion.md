# Tuned BM25 · weighted fusion

These Solutions combine the tuned, language-aware BM25 chunk retriever with one
semantic or learned-sparse branch. Every branch returns 50 whole-document
candidates from the benchmark's shared chunks. Scores are independently min-max
normalized per query before applying the fixed weights.

The representative weights are deliberately sparse rather than a Cartesian
product:

- BM25 / Qwen3 Embedding 8B: `40/60`
- BM25 / Voyage 4 Large: `20/80`
- BM25 / GTE Multilingual Base sparse: `50/50`

Weights were selected from a 0.05 grid and checked with five-fold out-of-fold
selection. Nearby weights produced similar results. The BM25 branch uses the
same per-language analyzers and scoring parameters as the standalone tuned BM25
Solution.

Milvus releases have exposed BM25 relevance scores with different numeric
directions. The fusion implementation therefore derives the direction from the
best-first ranking returned by the retriever, then normalizes the best score to
one and the worst score to zero.
