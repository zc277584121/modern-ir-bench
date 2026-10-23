# OpenSearch v2 Distill Doc-only Sparse

This Solution pins OpenSearch's compact English v2-distill document encoder. Documents receive neural sparse expansions, while queries use the checkpoint's static sparse encoder. This English-only Solution is evaluated on the complete English slice and is excluded from bilingual overall and Chinese rankings.

Documents use the shared short-context policy (384 target tokens, 448 maximum, 64 overlap), keeping every unit within the checkpoint's 512-token limit. Milvus performs sparse inner-product retrieval before hits are collapsed to whole document IDs. No BM25, dense embedding, reranking, or score fusion is added.
