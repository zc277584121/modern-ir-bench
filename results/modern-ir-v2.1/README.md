# Modern IR Bench v2.1 result evidence

This directory belongs to the benchmark result layer, not the Dataset.

`rankings.jsonl.gz` contains 7,000 immutable per-query outputs: seven published
Solutions across 1,000 queries. Each row records the Solution identifier, its Top-10
document ranking, and the per-query metrics used to verify the Space leaderboard.

The corresponding Dataset is `zc277584121/modern-ir-bench` at revision
`5bbc4ea34774b89ebec0e57ff82f231ff08d385c`. That Dataset contains only corpus,
queries, and binary Qrels.
