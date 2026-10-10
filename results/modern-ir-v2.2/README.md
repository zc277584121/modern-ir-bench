# Modern IR Bench v2.2 result evidence

This directory belongs to the benchmark result layer, not the Dataset.

v2.2 is a schema-only migration from v2.1: the Query field `task` was renamed to
`query_intent`. Query IDs and text, Corpus documents, Qrels, and the 7,000 saved
per-query Solution rankings are unchanged. Retrieval was not rerun.

The corresponding Dataset is `zc277584121/modern-ir-bench` at revision
`c7a38d6aaa3d1bd178952b6be91e12ca4ceb10ff` (tag `v2.2`).
