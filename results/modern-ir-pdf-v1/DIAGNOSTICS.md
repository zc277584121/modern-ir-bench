# Modern IR PDF Retrieval v1 fusion diagnostics

The fixed 25/75 score weights are compared with equal-weight reciprocal rank fusion on the same Top-50 branches.

| Retrieval branches | RRF NDCG@10 | Weighted NDCG@10 | Weighted delta |
| --- | ---: | ---: | ---: |
| OCR BM25 + BGE-M3 | 0.7703 | 0.6756 | -0.0947 |
| OCR BM25 + Voyage Multimodal 3.5 | 0.7961 | 0.7377 | -0.0585 |

The text benchmark's 25/75 lexical/dense weights do not transfer to PDF retrieval. Per-query min-max normalization does not make OCR lexical scores and visual or dense scores reliably comparable. RRF remains the recommended default for these heterogeneous PDF branches.

The weighted runs remain published as negative evidence. Detailed outcome counts and language slices are stored in `analysis.json`.
