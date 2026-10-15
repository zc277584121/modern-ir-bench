# Dense, BM25, and learned-sparse ablation

This study reuses frozen Top-50 route evidence. It performs no embedding, indexing, or retrieval calls. Weighted fusion uses per-query min-max normalization, a 0.05 simplex grid, and five-fold out-of-fold evaluation.

Dense baseline: Voyage 4 Large, NDCG@10 `0.8793`.

## bge-m3-sparse

| Configuration | RRF NDCG@10 | Weighted OOF NDCG@10 | Weighted EN | Weighted ZH | Full-data best weights (Dense/BM25/Sparse) |
| --- | ---: | ---: | ---: | ---: | --- |
| dense+bm25 | 0.8473 | 0.8886 | 0.8883 | 0.8889 | `0.75/0.25` |
| dense+learned_sparse | 0.8165 | 0.8812 | 0.8827 | 0.8797 | `0.90/0.10` |
| dense+bm25+learned_sparse | 0.8258 | 0.8886 | 0.8883 | 0.8889 | `0.75/0.25/0.00` |

Three-way weighted fold deltas versus Dense + BM25: `+0.0000`, `+0.0000`, `+0.0000`, `+0.0000`, `+0.0000`.

## gte-multilingual-sparse

| Configuration | RRF NDCG@10 | Weighted OOF NDCG@10 | Weighted EN | Weighted ZH | Full-data best weights (Dense/BM25/Sparse) |
| --- | ---: | ---: | ---: | ---: | --- |
| dense+bm25 | 0.8473 | 0.8886 | 0.8883 | 0.8889 | `0.75/0.25` |
| dense+learned_sparse | 0.8439 | 0.8862 | 0.8869 | 0.8856 | `0.80/0.20` |
| dense+bm25+learned_sparse | 0.8376 | 0.8882 | 0.8878 | 0.8886 | `0.80/0.10/0.10` |

Three-way weighted fold deltas versus Dense + BM25: `+0.0003`, `-0.0043`, `-0.0007`, `+0.0019`, `+0.0005`.

## Interpretation rule

Learned sparse has incremental fusion value only when Dense + BM25 + learned sparse beats Dense + BM25 out of fold. A zero selected sparse weight is evidence that the stored learned-sparse route adds no measurable value under this fusion method.
