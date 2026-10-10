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

Public, read-only benchmark explorer for the reviewed Modern IR Ranked Retrieval v2.2
release. The benchmark currently contains 5,000 bilingual synthetic documents, 1,000
queries, 6,575 positive binary Qrels, and seven retrieval Solutions.

v2.2 is a schema-only migration that renames the Query field `task` to `query_intent`.
The benchmark content, Qrels, rankings, and scores are unchanged from v2.1.

The current leaderboard is recomputed from a reviewed historical Top-10 ranking
artifact. The original run was not tied to a public Git commit, so result evidence and
maintained executable Solution definitions are linked separately.

- [Benchmark code and result evidence](https://github.com/zc277584121/modern-ir-bench)
- [Public Dataset](https://huggingface.co/datasets/zc277584121/modern-ir-bench)
