# PP-OCRv5 Mobile BM25 + Voyage Multimodal 3.5 · RRF

This hybrid combines two independent views of each PDF:

- PP-OCRv5 Mobile reads every page and BM25 retrieves complete OCR documents.
- Voyage Multimodal 3.5 embeds rendered page images directly, without OCR, and
  collapses page hits back to document IDs.

Each component retrieves its Top 50 documents. Equal-weight reciprocal rank
fusion combines the ranks with `k = 60`; only the final Top 10 is evaluated.
The method does not calibrate or mix incomparable lexical and dense scores.

This Solution is intended to measure whether visual evidence recovers relevant
documents missed by OCR text retrieval while BM25 preserves exact terms that a
visual embedding may overlook.
