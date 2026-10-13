# Candidate pool diagnostics

This report uses stored Top-50 route evidence. It does not run embedding, retrieval, or reranking.
Recall is averaged per query. Oracle NDCG@10 assumes a perfect reranker over the candidate set.

## modern-ir-ranked-retrieval-v2.2

1,000 queries · 6,575 positive Qrels

### Atomic routes

| Route | Recall@50 | Oracle NDCG@10 | All positives found | Unique positives |
| --- | ---: | ---: | ---: | ---: |
| BGE-M3 dense | 0.9598 | 0.9791 | 81.2% | 1 |
| Qwen3 Embedding 8B | 0.9708 | 0.9848 | 84.1% | 0 |
| Voyage 4 Large | 0.9911 | 0.9951 | 94.7% | 14 |
| BGE-M3 sparse | 0.9260 | 0.9560 | 71.9% | 0 |
| GTE Multilingual sparse | 0.9421 | 0.9669 | 76.2% | 0 |
| BM25 | 0.9167 | 0.9507 | 69.7% | 1 |

### Candidate unions

| Routes | Mean candidates | Relevant coverage | Oracle NDCG@10 | All positives found |
| --- | ---: | ---: | ---: | ---: |
| Voyage 4 Large + BM25 | 75.6 | 0.9973 | 0.9983 | 98.3% |
| Voyage 4 Large + GTE sparse | 73.1 | 0.9973 | 0.9985 | 98.4% |
| Voyage 4 Large + GTE sparse + BM25 | 89.4 | 0.9982 | 0.9989 | 99.0% |
| All stored routes | 120.7 | 0.9993 | 0.9997 | 99.6% |

### Pairwise complementarity

| Left | Right | Candidate Jaccard | Relevant only in left | Relevant only in right |
| --- | --- | ---: | ---: | ---: |
| BGE-M3 dense | Qwen3 Embedding 8B | 0.449 | 141 | 255 |
| BGE-M3 dense | Voyage 4 Large | 0.476 | 40 | 325 |
| BGE-M3 dense | BGE-M3 sparse | 0.342 | 451 | 205 |
| BGE-M3 dense | GTE Multilingual sparse | 0.360 | 374 | 235 |
| BGE-M3 dense | BM25 | 0.315 | 566 | 234 |
| Qwen3 Embedding 8B | Voyage 4 Large | 0.525 | 37 | 208 |
| Qwen3 Embedding 8B | BGE-M3 sparse | 0.330 | 530 | 170 |
| Qwen3 Embedding 8B | GTE Multilingual sparse | 0.356 | 426 | 173 |
| Qwen3 Embedding 8B | BM25 | 0.313 | 617 | 171 |
| Voyage 4 Large | BGE-M3 sparse | 0.355 | 585 | 54 |
| Voyage 4 Large | GTE Multilingual sparse | 0.384 | 475 | 51 |
| Voyage 4 Large | BM25 | 0.337 | 670 | 53 |
| BGE-M3 sparse | GTE Multilingual sparse | 0.567 | 122 | 229 |
| BGE-M3 sparse | BM25 | 0.391 | 358 | 272 |
| GTE Multilingual sparse | BM25 | 0.445 | 368 | 175 |

## modern-ir-pdf-retrieval-v1

1,000 queries · 6,575 positive Qrels

### Atomic routes

| Route | Recall@50 | Oracle NDCG@10 | All positives found | Unique positives |
| --- | ---: | ---: | ---: | ---: |
| OCR + BGE-M3 | 0.9405 | 0.9676 | 73.2% | 7 |
| Voyage Multimodal 3.5 | 0.9724 | 0.9861 | 85.0% | 40 |
| OCR + Voyage 4 Large | 0.9803 | 0.9902 | 87.9% | 27 |
| OCR + BM25 | 0.8948 | 0.9371 | 61.8% | 20 |

### Candidate unions

| Routes | Mean candidates | Relevant coverage | Oracle NDCG@10 | All positives found |
| --- | ---: | ---: | ---: | ---: |
| OCR Voyage 4 Large + Voyage Multimodal 3.5 | 66.8 | 0.9914 | 0.9959 | 93.9% |
| OCR Voyage 4 Large + OCR BM25 | 76.5 | 0.9893 | 0.9947 | 92.5% |
| OCR Voyage 4 Large + Voyage Multimodal 3.5 + OCR BM25 | 89.9 | 0.9956 | 0.9981 | 96.2% |
| All stored routes | 102.9 | 0.9964 | 0.9985 | 96.8% |

### Pairwise complementarity

| Left | Right | Candidate Jaccard | Relevant only in left | Relevant only in right |
| --- | --- | ---: | ---: | ---: |
| OCR + BGE-M3 | Voyage Multimodal 3.5 | 0.409 | 119 | 373 |
| OCR + BGE-M3 | OCR + Voyage 4 Large | 0.467 | 53 | 382 |
| OCR + BGE-M3 | OCR + BM25 | 0.299 | 620 | 278 |
| Voyage Multimodal 3.5 | OCR + Voyage 4 Large | 0.510 | 97 | 172 |
| Voyage Multimodal 3.5 | OCR + BM25 | 0.308 | 747 | 151 |
| OCR + Voyage 4 Large | OCR + BM25 | 0.321 | 754 | 83 |
