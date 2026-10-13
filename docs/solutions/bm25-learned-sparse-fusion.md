# BM25 + learned-sparse fusion

This Solution tests whether GTE Multilingual Base neural lexical expansion contributes complementary candidates beyond classic BM25 over the same 768-token chunks.

Each branch contributes its Top-50 whole-document ranking to equal-weight reciprocal rank fusion with `k=60`. The branch rankings and raw scores are retained in the local candidate cache for later analysis without rerunning either encoder.
