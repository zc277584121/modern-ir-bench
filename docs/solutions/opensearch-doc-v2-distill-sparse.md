# OpenSearch v2 Distill Doc-only Sparse

This Solution pins OpenSearch's compact English v2-distill document encoder. Documents receive neural sparse expansions, while queries use the checkpoint's static sparse encoder. The English-only model is intentionally evaluated on both language slices.

Milvus performs sparse inner-product retrieval over the shared chunks before hits are collapsed to whole document IDs. The checkpoint accepts up to 512 tokens, so its encoder truncates longer shared chunks. No BM25, dense embedding, reranking, or score fusion is added.
