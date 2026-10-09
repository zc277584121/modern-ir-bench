# Modern IR Bench v2.1 result evidence

This directory belongs to the benchmark result layer, not the Dataset.

`rankings.jsonl.gz` contains 7,000 immutable per-query outputs: seven published
Solutions across 1,000 queries. Each row records the Solution identifier, its Top-10
document ranking, and the per-query metrics used to verify the Space leaderboard.

The corresponding Dataset is `zc277584121/modern-ir-bench` at revision
`693f957b52713bcc564987f3ea5fb7883f438d51`. That Dataset contains only corpus,
queries, and binary Qrels.
