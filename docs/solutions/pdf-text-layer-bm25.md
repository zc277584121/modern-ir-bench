# Poppler text layer + BM25

This Solution is the lower bound for the PDF Retrieval Task.

## Pipeline

1. Receive each document as a Hugging Face `Pdf(decode=False)` value.
2. Run Poppler `pdftotext -layout` against the PDF path or bytes.
3. Preserve the document language declared by the Dataset.
4. Index the extracted text with language-aware Milvus BM25.
5. Return document-level rankings directly.

It does not run OCR and does not inspect rendered page images. An image-only PDF is
therefore indexed with an explicit empty-document placeholder. This is intentional:
the Solution measures what a conventional text-only PDF pipeline can recover and gives
OCR and visual Solutions a meaningful baseline to beat.

## Identity

- Public Solution ID: `poppler-text-layer-bm25-full`
- PDF extractor: `poppler-pdftotext-layout-v1`
- Retrieval: full-document Milvus BM25
- Candidate pools: independent Chinese and English pools
- Chunking: none
- OCR: none
- Primary metric: NDCG@10

The maintained Solution definition is
[`benchmarks/pdf_retrieval_solutions/poppler_bm25.py`](../../benchmarks/pdf_retrieval_solutions/poppler_bm25.py).
The public Task and Dataset binding is kept separately in
[`benchmarks/modern_ir_pdf_definition.py`](../../benchmarks/modern_ir_pdf_definition.py).
