# ColModernVBERT · page exact MaxSim

This English-only Solution encodes rendered PDF pages and text queries into normalized 128-dimensional token
vectors with the official Sentence Transformers `MultiVectorEncoder`. Exact MaxSim scores every page; the best
page score determines the document score. The index uses float16 dot products and float32 accumulation.

The adapter is pinned to `ModernVBERT/colmodernvbert@4c604428a6e1a7e33d0fb34837f0cdecbc22026e`, and its backbone to
`ModernVBERT/colmodernvbert-base@17604b47f51828a5e904557094552bf23fdd9fca`. A local adapter view binds the backbone
snapshot without modifying either official snapshot. GPU encoding uses bfloat16 and SDPA, with batch size 1 by
default (`MIR_COLMODERNVBERT_BATCH_SIZE`). Official processor settings and token masks are preserved. Outputs
are normalized in float32 before caching, matching the index's normalization and avoiding BF16 norm drift.
The official `1_Dense` projection contains the merged backbone projection and LoRA update; their weights and
bias were verified to match exactly. The upstream loader reports the original `custom_text_proj` keys as unused
because Sentence Transformers applies this separate Dense module.

The [official model card](https://huggingface.co/ModernVBERT/colmodernvbert) declares English support. Run with
`--language en`; Chinese and bilingual evaluation are rejected before encoding.

```bash
MIR_DEVICE=cuda:0 uv run --no-sync python -m benchmarks.pdf_retrieval.prepare \
  --solution colmodernvbert-page-exact-maxsim --language en
MIR_DEVICE=cuda:0 uv run --no-sync python -m benchmarks.pdf_retrieval.run \
  --solution colmodernvbert-page-exact-maxsim --language en --cached-only
```

Preparation persists both page and query embeddings, releases the encoder, and never opens an index. Scoring
requires a complete cache and cannot load a model to fill missing entries. Cache filenames contain a fingerprint
of the model, adapter/backbone revisions, dimension, normalization, and model/config/encode settings. Page keys
hash image content. Caches live in `.runtime/multi-vector/` and reuse the shared PDF page renderer.

Use `benchmarks.pdf_retrieval.smoke_visual_maxsim --solution colmodernvbert-page-exact-maxsim` for a real two-page,
two-query English check with isolated caches. See [the PDF guide](../pdf-solutions.md) for validation details.

The 2026-10-23 full run covers all 2,500 English documents, 5,702 pages, and 500 queries at Dataset revision
`5ac497e8aad79c9c0843c46eb97bb9900f3b109b`. The verified artifact pins source commit
`c826d3eea92eb90e1fd1f3de0265ec2820e1a8a9`; its rankings are published in the
[English release](../../results/modern-ir-pdf-v1/slices/en/).

| Scope | NDCG@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 0.61716885 | 0.13842397 | 0.45074518 | 0.61637215 | 0.78648730 |

This is rank **20 of 21** on the 2026-10-23 English PDF leaderboard by NDCG@10. It has no Overall or Chinese
score because the Solution's declared language coverage is English only.

On RTX 3080 Ti (`cuda:0`, batch size 1), cold embedding preparation took **2,693.94 seconds** and the
separate cache-only run took **64.11 seconds**, including cache reads, index construction, all 500 searches,
metrics, and artifact writing. The 2.077 GiB compressed cache contains every page and query input.
Pages retain 885–1,149 vectors each, totaling 5,223,678 page vectors; query lengths are 19–34 vectors.
The float16 page index occupies 1.245 GiB before allocator, coalescing, and scoring overhead.

Device memory sampled once per second peaked at 2,087 MiB during preparation and 4,273 MiB during scoring.
These device-wide observations include about 594 MiB held by the concurrent Jina process's default-device
CUDA context; the precision probe now selects the requested device explicitly. The preparation process exited
before scoring began, and the cache SHA-256, size, and mtime were unchanged after scoring. Artifact checksums,
500-query coverage, same-language candidate IDs, distinct Top-10 results, finite descending scores, and
recomputed metrics all passed verification.

The independent two-page smoke also passed official MaxSim and NumPy score checks (maximum error 0.001225),
unit-norm checks (maximum error 1.2e-7), and fresh-instance replay with model loading forbidden.
