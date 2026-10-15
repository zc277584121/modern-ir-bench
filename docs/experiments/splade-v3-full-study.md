# SPLADE v3 Full study

This experiment asks two narrow questions on the 500 English queries and 2,500
English documents in Modern IR Bench v2.2:

1. Does the full BERT-base SPLADE v3 route add value beyond Voyage 4 Large and
   tuned BM25?
2. Can SPLADE document expansion improve BM25 while retaining BM25's corpus-level
   IDF, term-frequency saturation, and length normalization?

The model is the pinned `castorini/splade-v3-onnx` export at revision
`51342029dd50ef0845c16081bd423039e7f61717`. Its model card declares the
`CC-BY-NC-SA-4.0` license. Documents use 384-target-token chunks, with 448 tokens
maximum and 64 tokens overlap. Every atomic route stores document-level Top-50
evidence before fusion.

## Atomic retrieval

| Route | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| Tuned BM25 | 0.7159 | 0.7075 |
| SPLADE v3 Full | 0.7450 | 0.7422 |
| Voyage 4 Large | 0.8832 | 0.8626 |

SPLADE v3 Full clearly improves on BM25 as a standalone English retriever, but
does not approach the strongest dense route on this dataset.

## Five-fold fusion ablation

Weighted fusion uses per-query min-max normalization, a 0.05 simplex grid, and
five-fold out-of-fold evaluation. The full-data weights are descriptive only;
the reported NDCG@10 is computed with weights selected without each held-out fold.

| Configuration | Weighted OOF NDCG@10 | Full-data weights |
| --- | ---: | --- |
| Voyage + BM25 | 0.8884 | 0.80 / 0.20 |
| Voyage + SPLADE | 0.8880 | 0.80 / 0.20 |
| Voyage + BM25 + SPLADE | 0.8882 | 0.75 / 0.10 / 0.15 |

The three-way fold deltas relative to Voyage + BM25 are `+0.0053`, `-0.0045`,
`-0.0054`, `+0.0006`, and `+0.0035`. SPLADE is complementary on some folds, but
the gain is not stable and the average is slightly lower.

## SPLADE expansion through BM25

For each document, SPLADE weights are max-aggregated across its short chunks.
The highest-weight novel alphabetic WordPieces are appended to the document,
then the same tuned English Milvus BM25 configuration (`k1=1.8`, `b=0.75`) builds
a new corpus index. Queries remain unchanged.

| Added terms per document | NDCG@10 | Delta | Recall@10 |
| ---: | ---: | ---: | ---: |
| 0 | 0.71591 | — | 0.70755 |
| 16 | 0.71659 | +0.00067 | 0.70860 |
| 32 | 0.71664 | +0.00073 | 0.70985 |
| 64 | 0.71160 | -0.00431 | 0.70473 |

The mechanism works, but the useful range is narrow and the gain is too small to
justify an additional published Solution. More expansion eventually adds enough
noise and document length to reduce quality.

## Decision

Keep the pinned runtime, candidate-cache command, ablation support, and expansion
experiment so the finding is reproducible. Do not add SPLADE v3 Full or expanded
BM25 to the public Space leaderboard. A future learned-sparse route should be
published only if it improves Voyage + tuned BM25 consistently out of fold, not
merely standalone BM25.
