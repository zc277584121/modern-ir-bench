# GTE sparse + Qwen3 Embedding 8B fusion

These Solutions combine the multilingual lexical weights from GTE Multilingual Base with dense Qwen3 Embedding 8B retrieval. Both branches search the same 768-token chunks and contribute a Top-50 whole-document ranking.

The RRF variant uses `k=60`. Weighted variants min-max normalize each branch per query and assign GTE-sparse/Qwen3-dense weights of 25/75, 50/50, or 75/25. Persisted branch evidence makes all weight sweeps offline and reproducible.
