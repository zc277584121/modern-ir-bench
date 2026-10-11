---
title: Modern IR Bench
emoji: 🔭
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Code-native benchmark for modern IR solutions
---

# Modern IR Bench

Public, read-only benchmark explorer for Modern IR Ranked Retrieval v2.2 and PDF
Retrieval v1. Both releases use the same 5,000 bilingual semantic documents, 1,000
queries, and 6,575 positive binary Qrels. PDF Retrieval replaces each text document
with a rendered PDF across varied layouts and capture conditions.

v2.2 is a schema-only migration that renames the Query field `task` to `query_intent`.
The benchmark content, Qrels, rankings, and scores are unchanged from v2.1.

The Ranked Retrieval leaderboard is recomputed from a reviewed historical Top-10
ranking artifact. The PDF baseline is a live run tied to pinned public code and data;
it intentionally reads only existing text layers and does not perform OCR.

- [Benchmark code and result evidence](https://github.com/zc277584121/modern-ir-bench)
- [Public Dataset](https://huggingface.co/datasets/zc277584121/modern-ir-bench)
- [Public PDF Dataset](https://huggingface.co/datasets/zc277584121/modern-ir-bench-pdf)
