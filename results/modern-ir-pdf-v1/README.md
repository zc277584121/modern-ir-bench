# Modern IR PDF Retrieval v1 result evidence

This directory preserves the first full, live result for the public PDF Retrieval Task.
It is result evidence, not a Dataset and not a Solution implementation.

The baseline extracts only existing PDF text layers with Poppler and runs full-document
Milvus BM25. It deliberately does not use OCR or page-image understanding, so image-only
relevant PDFs expose the baseline's expected failure boundary.

## Files

- `results.json`: aggregate metric records with immutable source provenance.
- `observations.parquet`: 1,000 query-level expected and Top-10 ranked document IDs.
- `manifest.json`: pinned Dataset revision and artifact checksums.
