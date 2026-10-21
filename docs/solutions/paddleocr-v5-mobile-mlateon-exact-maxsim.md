# PP-OCRv5 Mobile + mLateOn Exact MaxSim

This Solution measures multilingual late interaction after the same OCR stage
used by the lexical and dense PDF baselines.

## Pipeline

1. OCR every PDF page with the bilingual PP-OCRv5 mobile detector and recognizer.
2. Concatenate recognized text in page order and create overlapping 768-token chunks.
3. Encode every chunk with the pinned 307M-parameter `lightonai/mLateOn` revision,
   retaining its 128-dimensional token vectors.
4. Encode each query through mLateOn's query path.
5. On the GPU, score every chunk exhaustively: for each query token, take its
   maximum cosine similarity over all token vectors in that chunk, then sum the
   per-token maxima.
6. Keep the best chunk score for each PDF document and rank the documents.

The MaxSim stage does not use ANN candidate generation, TokenANN, PLAID, or an
index-specific pruning rule. It is a float32 exhaustive reference intended to
measure model quality without an approximate retrieval layer. The shared
model-independent chunker prevents model tokenizers from silently changing the
retrieval units. Embeddings are cached by model revision, role, and content so
repeated runs do not rerun the model.
