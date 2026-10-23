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
UV_PROJECT_ENVIRONMENT=.runtime/venvs/jina MIR_DEVICE=cuda:1 \
  uv run --no-sync python -m benchmarks.pdf_retrieval.prepare \
  --solution jina-v4-page-exact-maxsim
MIR_DEVICE=cuda:1 uv run --no-sync python -m benchmarks.pdf_retrieval.run \
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

The 2026-10-23 full run covers all 5,000 documents, 9,491 pages, and 1,000 queries at Dataset revision
`5ac497e8aad79c9c0843c46eb97bb9900f3b109b`. The verified scoring artifact pins source commit
`2731a090aa6c2a9739b8bc681e9edd575a9f4281`; its rankings are published in the
[bilingual release](../../results/modern-ir-pdf-v1/), with language slices derived from those same rankings.

| Scope | Queries | NDCG@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 1,000 | 0.80664184 | 0.18923948 | 0.61093415 | 0.78573093 | 0.93172063 |
| English | 500 | 0.81840212 | 0.19179963 | 0.62315122 | 0.80173848 | 0.93605794 |
| Chinese | 500 | 0.79488156 | 0.18667934 | 0.59871709 | 0.76972338 | 0.92738333 |

By NDCG@10 on the 2026-10-23 PDF leaderboard, these scores rank **4/16 Overall, 4/21 English, and 5/16 Chinese**.
Jina leads the published OCR-free visual Solutions in all three scopes. Compared with ColQwen2.5, its NDCG@10
is higher by 0.03105667 Overall, 0.02011034 English, and 0.04200301 Chinese.

On RTX 3080 Ti (`cuda:1`, batch size 1), cold preparation in the isolated Jina environment took
**4,429.43 seconds**. The separate cache-only run in the main environment took **81.15 seconds**, including
cache reads, both language indexes, all 1,000 searches, metrics, and artifact writing. Device memory sampled
once per second peaked at **9,576 MiB** during preparation and **2,906 MiB** during scoring. Device usage
returned to 4 MiB between the processes, confirming that encoding weights were unloaded.

The compressed page/query cache occupies 2.881 GiB. English pages contain 4,266,502 vectors (725–779 per page),
and Chinese pages 2,836,005 vectors (725–770 per page). The separate float16 indexes occupy 1.017 GiB and
0.676 GiB before allocator and scoring overhead. Query lengths are 9–21 English and 10–24 Chinese vectors.
All input keys are present, and cache SHA-256, size, and mtime remained unchanged during scoring.
Artifact checksums, 500-query coverage per language, same-language candidate IDs, distinct Top-10 results,
finite descending scores, and recomputed metrics all passed verification.

The independent bilingual smoke also passed official-formula and NumPy score checks (maximum error 0.000966),
unit-norm checks (maximum error 1.2e-7), and fresh-instance replay with model loading forbidden. Its cache was
also replayed successfully under the main environment's Transformers 5.19.

```bash
UV_PROJECT_ENVIRONMENT=.runtime/venvs/jina MIR_DEVICE=cuda:0 \
  uv run --no-sync python -m benchmarks.pdf_retrieval.smoke_visual_maxsim \
  --solution jina-v4-page-exact-maxsim --output .runtime/diagnostics/jina-v4-smoke.json
```
