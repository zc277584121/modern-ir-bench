# PP-OCRv5 Mobile + BGE-M3

This is the local dense counterpart to the OCR + BM25 baseline.

## Pipeline

1. OCR every PDF page with the bilingual PP-OCRv5 mobile detector and recognizer.
2. Concatenate recognized text in page order.
3. Encode each complete document with the pinned `BAAI/bge-m3` revision.
4. Encode queries through BGE-M3's query path.
5. Search exact cosine similarity with a Milvus FLAT index.

The Solution shares OCR output with the lexical baseline, so the comparison
isolates lexical versus dense retrieval after identical document extraction. It
runs locally, uses 1024-dimensional vectors, and does not use the PDF text layer.
