# BM25 + Voyage 4 Large · RRF

This hybrid retrieves the shared 768-token chunks independently with BM25 and
Voyage 4 Large, then collapses each branch to a Top-50 whole-document ranking.
Voyage document and query embeddings use the official Voyage API at 1024
dimensions.

Equal-weight reciprocal rank fusion combines the two rankings with `k=60`.
The branch rankings and scores are retained locally, allowing alternative
weights to be evaluated in Fusion Lab without another API or retrieval call.
