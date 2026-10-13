# GTE sparse + Qwen3 Embedding 8B fusion

This Solution combines the multilingual lexical weights from GTE Multilingual Base with dense Qwen3 Embedding 8B retrieval. Both branches search the same 768-token chunks and contribute a Top-50 whole-document ranking.

The published Solution uses equal-weight reciprocal rank fusion with `k=60`. Persisted branch evidence makes alternative weights and fusion methods available interactively in Fusion Lab without rerunning either model.
