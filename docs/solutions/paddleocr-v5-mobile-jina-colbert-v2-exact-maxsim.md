# PP-OCRv5 Mobile + Jina-ColBERT-v2 Exact MaxSim

This Solution evaluates the multilingual `jinaai/jina-colbert-v2` checkpoint
after bilingual PP-OCRv5 Mobile text extraction.

## Pipeline

1. OCR every PDF page and concatenate the recognized text in page order.
2. Create the benchmark's standard overlapping 768-token chunks with its
   pinned, model-independent tokenizer. Override the checkpoint artifact's
   legacy 300-token document cap with a 4,096-token cap, which covers the
   longest Jina-tokenized chunk in this Dataset without truncation.
3. Encode each chunk and query with the pinned 0.6B-parameter
   `jinaai/jina-colbert-v2` revision, retaining its 128-dimensional token vectors.
4. Score every chunk exhaustively in float32 with MaxSim on the GPU.
5. Keep the best chunk score for each PDF document and rank the documents.

Jina-ColBERT-v2 supports 94 languages and an 8192-token input context, so both
the Chinese and English Dataset slices are evaluated. The retrieval stage does
not use ANN candidate generation, TokenANN, PLAID, or index-specific pruning.
The model checkpoint is distributed under CC BY-NC 4.0; its public benchmark
result is therefore an evaluation result, not a claim that the model is
available for unrestricted commercial use.
