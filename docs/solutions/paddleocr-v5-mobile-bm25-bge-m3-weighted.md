# PP-OCRv5 Mobile + BM25/BGE-M3 · Weighted 25/75

PP-OCRv5 Mobile reads every PDF page once. Two full-document retrieval paths
then operate on the same extracted text:

- Milvus BM25 preserves exact bilingual terms.
- BGE-M3 supplies multilingual dense similarity.

Each path returns 50 documents. Scores are normalized independently for every
query with min-max normalization, combined with 25% BM25 and 75% BGE-M3, and
reduced to the final Top 10. The fixed weights match the text benchmark's
predeclared lexical/dense balance; they are not tuned on the PDF Qrels.
