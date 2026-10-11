# Modern IR PDF Retrieval v1 result evidence

This directory preserves the first full, live result for the public PDF Retrieval Task.
It is result evidence, not a Dataset and not a Solution implementation.

The baseline OCRs every page with the bilingual PP-OCRv5 Mobile detector and recognizer,
then runs full-document Milvus BM25. It never reads native PDF text layers and does not
use layout parsing, chunking, dense embeddings, or reranking.

## Files

- `results.json`: aggregate metric records with immutable source provenance.
- `observations.parquet`: 1,000 query-level expected and Top-10 ranked document IDs.
- `manifest.json`: pinned Dataset revision and artifact checksums.
