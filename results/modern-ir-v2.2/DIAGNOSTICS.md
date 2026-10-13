# Modern IR Ranked Retrieval v2.2 diagnostics

This report is derived from the frozen Top-10 release and the exported Top-50 candidate evidence.
It diagnoses candidate coverage, reranker behavior, and benchmark difficulty without changing Qrels.

## Candidate ceiling

| Candidate path | Recall@50 | Any relevant | All relevant |
| --- | ---: | ---: | ---: |
| `qwen3-embedding-8b` | 0.9708 | 100.0% | 84.1% |
| `bm25-qwen3-embedding-8b-weighted-25-75` | 0.9825 | 100.0% | 89.5% |

## Reranker effect

- **Qwen3 Embedding 8B → Voyage Rerank 3**: ΔNDCG@10 `+0.1233`; 702 improved, 86 regressed, 212 unchanged.
- **Weighted candidate path → Voyage Rerank 3**: ΔNDCG@10 `+0.0974`; 657 improved, 105 regressed, 238 unchanged.
- **Qwen3 Embedding 8B → BGE Reranker v2 M3**: ΔNDCG@10 `+0.0290`; 492 improved, 382 regressed, 126 unchanged.
- **Weighted candidate path → BGE Reranker v2 M3**: ΔNDCG@10 `-0.0021`; 415 improved, 431 regressed, 154 unchanged.

## Difficulty signal

| Solution | Relevant at rank 1 | All relevant in Top 10 |
| --- | ---: | ---: |
| `qwen3-embedding-8b-rerank-3` | 99.2% | 59.9% |
| `bm25-qwen3-embedding-8b-weighted-rerank-3` | 99.3% | 61.5% |

The first positive is easy for the strongest reranker, but complete positive-set recall is not. NDCG@10 and Recall@10 should therefore be read together; MRR alone overstates saturation.

Detailed slices and concrete best/worst queries are stored in `analysis.json`.
Top-50 evidence is stored in `candidate-rankings.jsonl.gz`; its checksum is recorded in `analysis.json`.
