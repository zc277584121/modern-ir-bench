# PP-OCRv5 Mobile + Voyage 4 Large

This Solution keeps the PDF understanding stage identical to the BM25 baseline
and changes only retrieval.

## Pipeline

1. OCR every PDF page with the bilingual PP-OCRv5 mobile detector and recognizer.
2. Concatenate recognized text in page order.
3. Encode each complete document with `voyageai/voyage-4-large` through
   OpenRouter's Embeddings API at 1024 dimensions.
4. Encode text queries with the same model's query input type.
5. Search exact cosine similarity with a Milvus FLAT index.

Using the same OCR cache as the lexical baseline isolates the effect of the
retrieval representation. It does not use the native PDF text layer, layout
parsing, chunking, or reranking.

Embeddings are cached by model, dimension, input type, and exact OCR text so an
interrupted run can resume without repeating completed API calls.
