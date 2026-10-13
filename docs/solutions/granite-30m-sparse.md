# Granite Embedding 30M Sparse

This Solution pins IBM's 30M-parameter English sparse bi-encoder as a deliberately small learned-sparse baseline. It generates weighted vocabulary expansions for both queries and documents and uses the checkpoint's native 512-token limit.

Milvus performs sparse inner-product retrieval over the shared chunks before results are collapsed to whole document IDs. The full bilingual evaluation makes the quality and language-coverage cost of this very small model visible.
