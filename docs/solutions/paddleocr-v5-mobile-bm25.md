# PP-OCRv5 Mobile + BM25

This Solution is the simple OCR baseline for the PDF Retrieval Task.

## Pipeline

1. Receive each document as a Hugging Face `Pdf(decode=False)` value.
2. Render every page through PaddleOCR's PDF input path.
3. Detect and recognize all pages with `PP-OCRv5_mobile_det` and
   `PP-OCRv5_mobile_rec`.
4. Concatenate recognized lines in page order.
5. Index each complete OCR document with language-aware Milvus BM25.
6. Return document-level rankings directly.

The Solution never reads the native PDF text layer. A born-digital PDF and an
image-only scan therefore pass through exactly the same OCR path. It deliberately
does not use orientation classification, image rectification, layout parsing,
chunking, dense embeddings, or reranking.

OCR output is cached by PDF content hash so an interrupted benchmark can resume.
The cache changes runtime only; it does not use Dataset labels or affect rankings.

## Identity

- Public Solution ID: `paddleocr-v5-mobile-bm25-full`
- Detection: `PP-OCRv5_mobile_det`
- Recognition: `PP-OCRv5_mobile_rec`
- Languages: Chinese and English with one recognition model
- Retrieval: full-document Milvus BM25
- Candidate pools: independent Chinese and English pools
- Chunking: none
- Primary metric: NDCG@10

The maintained Solution definition is
[`benchmarks/pdf_retrieval_solutions/paddleocr_bm25.py`](../../benchmarks/pdf_retrieval_solutions/paddleocr_bm25.py).
The public Task and Dataset binding is kept separately in
[`benchmarks/modern_ir_pdf_definition.py`](../../benchmarks/modern_ir_pdf_definition.py).
