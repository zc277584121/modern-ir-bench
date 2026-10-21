# PP-OCRv5 Mobile + ColBERTv2 Exact MaxSim

This Solution evaluates the canonical English `colbert-ir/colbertv2.0`
checkpoint after bilingual PP-OCRv5 Mobile text extraction. Only the English
Dataset slice is valid for this English-only model.

## Pipeline

1. OCR every PDF page and concatenate the recognized text in page order.
2. Use the benchmark's pinned, model-independent tokenizer to create short
   chunks targeting 128 tokens, capped at 160 tokens, with 32-token overlap.
   This stays within ColBERTv2's native 180-token document limit without
   silently truncating the benchmark's standard 768-token chunks.
3. Encode each chunk and query with the pinned 110M-parameter
   `colbert-ir/colbertv2.0` revision, retaining its 128-dimensional token vectors.
4. Score every chunk exhaustively in float32 with MaxSim on the GPU.
5. Keep the best chunk score for each PDF document and rank the documents.

The retrieval stage does not use ANN candidate generation, TokenANN, PLAID, or
index-specific pruning. The checkpoint is MIT licensed.
