# PP-OCRv5 Mobile + mLateOn MaxSim

This Solution measures multilingual late interaction after the same OCR stage
used by the lexical and dense PDF baselines.

## Pipeline

1. OCR every PDF page with the bilingual PP-OCRv5 mobile detector and recognizer.
2. Concatenate recognized text in page order and create overlapping 768-token chunks.
3. Encode every chunk with the pinned 307M-parameter `lightonai/mLateOn` revision,
   retaining its 128-dimensional token vectors.
4. Encode each query through mLateOn's query path.
5. Search the chunk matrices with Milvus `MAX_SIM_COSINE`, then keep the best
   chunk score for each PDF document.

The shared model-independent chunker prevents model tokenizers from silently
changing the retrieval units. Embeddings are cached by model revision, role,
and content so index experiments do not rerun the model.
