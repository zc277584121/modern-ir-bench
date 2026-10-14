# OpenSearch Multilingual v1 Sparse

This Solution pins OpenSearch's 160M multilingual learned-sparse model. Query encoding uses the model's static token-weight route, while document encoding performs neural expansion. Both Chinese and English are evaluated in the same bilingual release.

Documents use the shared short-context policy (384 target tokens, 448 maximum, 64 overlap), keeping every unit within the checkpoint's 512-token limit. Milvus stores the sparse vectors in a `SPARSE_INVERTED_INDEX` and retrieves them with inner product before chunk hits are collapsed to whole document IDs. The model is used without additional pruning so this run measures its full released representation.
