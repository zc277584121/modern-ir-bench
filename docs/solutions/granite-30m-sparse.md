# Granite Embedding 30M Sparse

This Solution pins IBM's 30M-parameter English sparse bi-encoder as a deliberately small learned-sparse baseline. It generates weighted vocabulary expansions for both queries and documents and uses the checkpoint's native 512-token limit.

Documents use the shared short-context policy (384 target tokens, 448 maximum, 64 overlap), so the encoder does not silently truncate the benchmark's normal 768-token chunks. Following the model card, query vectors retain the 50 largest dimensions, document vectors retain 192, and Milvus uses a 0.2 drop ratio for index construction and search.

Milvus performs sparse inner-product retrieval before chunk results are collapsed to whole document IDs. The full bilingual evaluation makes the quality and language-coverage cost of this very small model visible.
