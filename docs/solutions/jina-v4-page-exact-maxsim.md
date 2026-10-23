# Jina Embeddings v4 · page exact MaxSim

This bilingual Solution uses Jina v4's native retrieval multi-vector head to encode rendered PDF pages and text
queries into normalized 128-dimensional token vectors. Exact MaxSim scores every page and retains the best page
score per document. The index uses float16 dot products and float32 accumulation.

The complete model snapshot, native code, and retrieval adapters are pinned to
`jinaai/jina-embeddings-v4@853c867b65b749f3c3c72a06868140d842e04f06`. The thin adapter calls
`encode_text(task="retrieval", prompt_name="query", return_multivector=True)` and
`encode_image(task="retrieval", return_multivector=True)`. Native attention masks remove padding before caching.
It uses bfloat16 and SDPA on supported GPUs, at most 602,112 image pixels, a 32,768-token text budget, and batch size 1
by default (`MIR_JINA_V4_BATCH_SIZE`). The local pinned snapshot also fixes the native loader's nested adapter load.
Outputs are normalized in float32 before caching, matching the index's normalization and avoiding BF16 norm drift.

The [official model card](https://huggingface.co/jinaai/jina-embeddings-v4) documents the native interface;
the [technical report](https://arxiv.org/html/2506.18902v3) includes English and Chinese visual multi-vector results.
Native encoding uses the isolated `jina` dependency group: Transformers 4.57.6 and PEFT 0.18.1. The pinned
processor cannot initialize under Transformers 5.19, which remains the runtime for the Sentence Transformers
Solutions. Both encoding dependencies are included in Jina's cache fingerprint. Cached scoring needs no native
model loader and works in the main environment.

```bash
UV_PROJECT_ENVIRONMENT=.runtime/venvs/jina uv sync --group jina --group multimodal --group pdf
UV_PROJECT_ENVIRONMENT=.runtime/venvs/jina MIR_DEVICE=cuda:0 \
  uv run --no-sync python -m benchmarks.pdf_retrieval.prepare \
  --solution jina-v4-page-exact-maxsim
MIR_DEVICE=cuda:0 uv run --no-sync python -m benchmarks.pdf_retrieval.run \
  --solution jina-v4-page-exact-maxsim --cached-only
```

Preparation persists both page and query embeddings without retrieval and releases the model before scoring.
The default scoring path requires a complete cache; a missing entry fails with preparation instructions.
Configuration-addressed cache filenames include the pinned revision, retrieval task, query prompt, multi-vector
mode, precision, attention implementation, input budgets, and dimension. Page keys hash image content. SQLite
stores float32 matrices with lossless compression under `.runtime/multi-vector/`.

The real smoke covers two English and two Chinese pages and queries using isolated caches. It checks shapes,
normalization, finite values, numerical MaxSim agreement, semantic rankings within each language pool, and fresh
encoder replay with model loading forbidden. The current native API has no scoring method; the independent
reference reproduces Jina's published padded einsum/max/sum scorer at
[`d347c952`](https://huggingface.co/jinaai/jina-embeddings-v4/blob/d347c952d22e0aabbeabe2a28c205c6c3d5dfaad/modeling_colqwen_duo.py)
and also checks a normalized NumPy reference. See [the PDF guide](../pdf-solutions.md).

The 2026-10-23 RTX 3080 Ti smoke encoded four pages with 779 vectors each and queries with 10/8 English
and 12/12 Chinese vectors. Cold encoding and cache writes took 9.66 seconds, including loading and rendering;
the complete check took 11.25 seconds. Peak CUDA allocation/reservation was 8.178/8.383 GiB during encoding
and 9.08/30 MiB during scoring and replay. Device memory sampled every 200 ms peaked at 8,904 MiB.
Maximum official-score error was 0.000966 and norm error was 1.2e-7; all four queries ranked their matching
page first. The isolated embedding cache occupied 1.328 MiB. A separate replay under the main environment's
Transformers 5.19 took 0.38 seconds with model loading forbidden and unchanged cache SHA-256 and mtime.
These measurements cover tiny synthetic pages, not full-dataset throughput or index memory.

```bash
UV_PROJECT_ENVIRONMENT=.runtime/venvs/jina MIR_DEVICE=cuda:0 \
  uv run --no-sync python -m benchmarks.pdf_retrieval.smoke_visual_maxsim \
  --solution jina-v4-page-exact-maxsim --output .runtime/diagnostics/jina-v4-smoke.json
```
