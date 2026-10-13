# Modern IR PDF Retrieval v1 diagnostics

Published RRF Solutions are compared with their independently published component baselines.
All fusion routes contribute Top-50 candidates and use equal-weight RRF with `k=60`.

| Retrieval routes | Fusion NDCG@10 | Best component NDCG@10 | Delta |
| --- | ---: | ---: | ---: |
| OCR BM25 + BGE-M3 | 0.7703 | 0.7277 | +0.0426 |
| OCR BM25 + Voyage Multimodal 3.5 | 0.7961 | 0.7922 | +0.0039 |
| OCR Voyage 4 Large + Voyage Multimodal 3.5 | 0.8442 | 0.8499 | -0.0057 |

Alternative route weights remain exploratory and can be computed from stored Top-50 evidence in Fusion Lab.
They are not published as separate leaderboard Solutions.

## Reranker effect

| Candidate route | Before NDCG@10 | After NDCG@10 | Delta |
| --- | ---: | ---: | ---: |
| OCR Voyage 4 Large + Voyage Multimodal 3.5 → Voyage Rerank 3 | 0.8442 | 0.9196 | +0.0754 |

The reranker receives only the final Top-50 documents from the fused candidate route.
