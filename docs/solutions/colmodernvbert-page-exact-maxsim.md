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

The 2026-10-23 RTX 3080 Ti smoke used 884 vectors per page and 21/18 vectors per query. Cold preparation took
3.85 seconds, including loading, rendering, and cache writes; the complete check took 6.33 seconds. Peak CUDA
allocation/reservation was 0.847/0.941 GiB during encoding and 9.21/80 MiB during scoring and replay. Maximum
official-score error was 0.001225 and norm error was 1.2e-7. Both queries ranked their matching page first,
and a fresh encoder replayed the 0.746 MiB cache with model loading forbidden. These tiny synthetic pages do not
measure full-dataset throughput or peak memory.
