# PP-OCRv5 Mobile BM25 + Voyage Multimodal 3.5 · Weighted 25/75

This cross-modal hybrid combines two independent PDF views:

- PP-OCRv5 Mobile extracts complete document text for Milvus BM25 retrieval.
- Voyage Multimodal 3.5 embeds rendered page images and collapses page hits to
  document IDs.

Each path returns 50 documents. Scores are normalized independently for every
query with min-max normalization, combined with 25% OCR/BM25 and 75% visual
embedding weight, and reduced to the final Top 10. The fixed weights are
declared in code and are not tuned on the PDF Qrels.
