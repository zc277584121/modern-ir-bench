# OpenSearch Multilingual v1 Sparse

This Solution pins OpenSearch's 160M multilingual learned-sparse model and encodes the shared 768-token chunks into weighted vocabulary entries. Query encoding uses the model's static token-weight route, while document encoding performs neural expansion. Both Chinese and English are evaluated in the same bilingual release.

Milvus stores the sparse vectors in a `SPARSE_INVERTED_INDEX` and retrieves them with inner product before chunk hits are collapsed to whole document IDs. The checkpoint accepts up to 512 tokens, so its encoder truncates longer shared chunks. The model is used without additional pruning so this run measures its full released representation.
