# GTE Multilingual Base Sparse

This Solution uses the sparse token-weight output of the pinned 304M GTE Multilingual Base checkpoint. It applies the checkpoint's learned non-negative weight to each input token, keeps the strongest weight for repeated vocabulary IDs, and evaluates Chinese and English with the same model.

The shared 768-token chunks are encoded locally, indexed in Milvus with sparse inner-product search, and collapsed back to whole document IDs. This is the sparse branch only; no dense score is fused into the result.
