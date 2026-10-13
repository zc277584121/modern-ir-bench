# PDF Voyage Rerank 3

This two-stage Solution combines OCR-based text retrieval with OCR-free page-image retrieval. Each route contributes a Top-50 document ranking to equal-weight RRF. The fused ranking is truncated to 50 documents before reranking, so increasing the number of routes never increases the reranker input beyond the benchmark limit.

Voyage `rerank-3` receives the query and the complete PP-OCRv5 Mobile text of each candidate PDF. It replaces the RRF order with relevance scores and returns the best ten document ids. Candidate rankings and exact query-document reranker scores are cached locally; the caches are execution artifacts and are not published.
