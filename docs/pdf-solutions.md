# PDF Retrieval Solution Plan

The PDF Task always evaluates document identifiers. A Solution may read a PDF in
any way, but page or chunk hits must be collapsed back to unique document IDs
before scoring.

## Maintained catalog

| Solution | PDF reading | Retrieval unit | Representation | Purpose |
| --- | --- | --- | --- | --- |
| PP-OCRv5 Mobile + BM25 | OCR | whole document | lexical | portable baseline |
| PP-OCRv5 Mobile + BGE-M3 | OCR | whole document | local dense | isolate dense versus lexical retrieval |
| PP-OCRv5 Mobile + BM25/BGE-M3 RRF | OCR | whole document | local hybrid | combine exact terms and semantic matches |
| PP-OCRv5 Mobile + Voyage 4 Large | OCR | whole document | hosted dense | compare a stronger hosted text model |
| Voyage Multimodal 3.5 | rendered images | page | visual dense | test OCR-free retrieval |
| EmbeddingGemma 2 | rendered images | page | local visual dense | test a lightweight native multimodal model |
| WeMM Embedding 2B | rendered images | page | local visual dense | test an open bilingual visual-document model |
| Gemini Embedding 2 | rendered images | page | hosted visual dense | test a general managed multimodal embedding model |
| Llama Nemotron Embed VL 1B V2 | rendered images | page | hosted visual dense | test a compact managed visual-language embedding model |
| Ovis Omni Embedding 3B | rendered images | page | local visual dense | test a self-hosted bilingual omni-modal embedding model |
| ColSmol 256M / 500M | rendered images | page | local visual multi-vector | compare compact visual late-interaction models |
| ColPali v1.3 | rendered images | page | local visual multi-vector | evaluate PaliGemma-based late interaction |
| ColQwen2 v1.0 / ColQwen2.5 v0.2 | rendered images | page | local visual multi-vector | evaluate Qwen-based late interaction |
| ColModernVBERT | rendered images | page | local visual multi-vector | compact English-only late interaction |
| Jina Embeddings v4 | rendered images | page | native visual multi-vector | bilingual retrieval with the 128-dimensional head |
| PP-OCRv5 Mobile BM25 + Voyage Multimodal 3.5 RRF | OCR + rendered images | document + page | lexical + visual dense | combine exact text and visual evidence |
| OCR Voyage 4 Large + Voyage Multimodal 3.5 RRF | OCR + rendered images | document + page | text dense + visual dense | combine official Voyage text and visual routes |
| OCR Voyage 4 Large + Voyage Multimodal 3.5 → Voyage Rerank 3 | OCR + rendered images | whole document | fused retrieval + reranking | rerank the final Top-50 fused candidates from complete OCR text |

All dense indexes use exact cosine search through Milvus FLAT. Visual multi-vector
Solutions use the shared Torch Exact MaxSim index. Every visual Solution keeps
the best page score for each document.

RRF is the published default for heterogeneous PDF branches. Alternative
per-query normalized score weights can be evaluated interactively from stored
Top-50 route evidence, so they do not need separate leaderboard entries.

The reranking Solution keeps that search space bounded: each route contributes at
most Top-50, RRF produces one final Top-50 candidate list, and Voyage Rerank 3
returns the published Top-10 document ranking.

## Next controlled additions

1. **PP-OCRv5 Server + BM25/BGE-M3** changes only OCR quality. It should be added
   after the current OCR and dense baselines have full results because it requires
   a second complete OCR pass.

Native PDF text extraction is deliberately excluded: image-only and damaged-text
PDFs would silently take a different path, which would make it a format shortcut
rather than a valid baseline for this Dataset.

## Runtime and caches

- OCR text is cached by PDF content hash.
- Rendered pages are cached by renderer configuration and PDF content hash.
- Hosted embeddings are cached by model, input type, dimension, and input content.
- Voyage Multimodal uses batches of 32 pages by default. This keeps image upload
  bodies manageable even though the API token limit would permit larger batches.
  Override it with `MIR_VOYAGE_MULTIMODAL_BATCH_SIZE` when project limits and
  network conditions differ.
- Two multimodal requests may run concurrently so network and service latency do
  not serialize the full indexing pass. Set `MIR_VOYAGE_MULTIMODAL_CONCURRENCY`
  to match the active project RPM limit.
- OpenRouter visual embeddings share the same content-addressed cache and expose
  batch and concurrency controls through `MIR_OPENROUTER_IMAGE_BATCH_SIZE` and
  `MIR_OPENROUTER_IMAGE_CONCURRENCY`.
- Ovis Omni Embedding 3B runs locally in four-page batches by default. Override
  this with `MIR_OVIS_BATCH_SIZE` for smaller GPUs.
- Milvus-backed Solutions create isolated collections and drop them after the
  evaluation session closes. Exact MaxSim Solutions keep their index in Torch
  memory and release it when the session closes.

Voyage 4 Large text embeddings and Voyage Multimodal 3.5 page embeddings use
Voyage's official API. Other hosted visual models retain the provider path
recorded by their source commit.

## Visual Exact MaxSim

Visual late-interaction Solutions share `CachedMultiVectorEmbedding`,
`ChunkedMultiVectorRetrievalSolution`, and `TorchExactMaxSimIndex`. Sentence Transformers configurations
define preprocessing, prompts, projections, and token masks for ColSmol, ColPali, ColQwen, and ColModernVBERT.
Jina v4 uses its native retrieval multi-vector API. Model revisions are pinned in `benchmarks/model_settings.py`;
ColModernVBERT also binds its adapter to an explicitly pinned backbone snapshot.
The Sentence Transformers models use the main benchmark environment:

```bash
uv sync --group benchmark --group multimodal --group pdf --group dev
```

Jina's native processor requires a separate dependency group with Transformers 4.57.6 and PEFT 0.18.1;
its constructor is incompatible with the main environment's Transformers 5.19. Keep both environments:

```bash
UV_PROJECT_ENVIRONMENT=.runtime/venvs/jina uv sync --group jina --group multimodal --group pdf --group dev
```

Set `UV_PROJECT_ENVIRONMENT=.runtime/venvs/jina` for Jina preparation and smoke. Cache-only scoring can run
in the main environment. Model snapshots and persistent embeddings are shared; no model-code patches are used.

The smoke uses two PDFs and queries per supported language. It exercises the real renderer and model,
compares Exact MaxSim against the official scorer and a normalized NumPy calculation, and verifies fresh
encoder cache replay with model loading forbidden. Jina's current API has no score method; its reference
reproduces the published padded einsum/max/sum function linked in its Solution README. Score checks use
float32 reference arithmetic on unit-normalized inputs, matching the index, and an absolute tolerance of 0.02
for float16 scoring. This also supports existing caches containing native BF16 normalization drift.
Inputs and embedding caches are temporary unless `--work-dir` retains them in a separate directory;
the optional JSON report records encoding/scoring timings, first-batch timings, shapes, errors, and separate GPU peaks.
Retained smoke runs require an empty embedding cache so their encoding measurements represent cold preparation.
Bilingual smokes require a Chinese-capable font (`--font`) for Chinese pages; English uses a separate Latin font.
This command never loads the benchmark Dataset.

```bash
MIR_DEVICE=cuda:0 uv run --no-sync python -m benchmarks.pdf_retrieval.smoke_visual_maxsim \
  --solution colsmol-500m-page-exact-maxsim \
  --output .runtime/diagnostics/colsmol-500m-smoke.json
```

Use `colpali-v1-3-page-exact-maxsim` or `colqwen2-v1-0-page-exact-maxsim` to check the other new models.
For `HF_HUB_OFFLINE=1`, cache complete official model snapshots, including metadata and named chat
templates, along with their adapter backbones. The Hub checks snapshot completeness when discovering
cached templates. Run the smoke once with network access so the official loader also caches revision
references and missing-file metadata before switching to offline mode.
Default encoding batches are 4 for ColSmol 500M and 1 for ColPali v1.3 and ColQwen2 v1.0. Override them
with `MIR_COLSMOL_500M_BATCH_SIZE`, `MIR_COLPALI_V1_3_BATCH_SIZE`, or `MIR_COLQWEN2_V1_0_BATCH_SIZE`.

The full benchmark records source provenance and requires a clean Git worktree. Commit reviewed changes
before running it. ColSmol 256M/500M and ColPali v1.3 support only the English slice
in this benchmark; ColQwen2 and ColQwen2.5 support both languages. To evaluate
three Solutions sequentially on the English slice using one GPU:

```bash
MIR_DEVICE=cuda:0 \
MIR_COLSMOL_500M_BATCH_SIZE=1 \
MIR_COLPALI_V1_3_BATCH_SIZE=1 \
MIR_COLQWEN2_V1_0_BATCH_SIZE=1 \
uv run --no-sync python -m benchmarks.pdf_retrieval.run \
  --language en \
  --solution colsmol-500m-page-exact-maxsim \
  --solution colpali-v1-3-page-exact-maxsim \
  --solution colqwen2-v1-0-page-exact-maxsim
```

These routes do not require a running Milvus service or OCR. Page embeddings persist under
`.runtime/multi-vector/`; rendered pages reuse `.runtime/pdf-pages/pdfium-v1/`. Run artifacts go to
`artifacts/modern-ir-pdf-runs/`, with language runs under `slices/<language>/`.

## Prepared embeddings and cache-only scoring

ColModernVBERT (English only) and Jina v4 (English and Chinese) default to cache-only scoring. First prepare
all page and query embeddings, then run the scoring process. Preparation opens no index, computes no rankings,
and writes no benchmark result. It commits each encoding batch, supports resuming an interrupted cache, and
releases model weights before returning. The two phases use the same page renderer and input transforms.

```bash
MIR_DEVICE=cuda:0 NUMPY_MADVISE_HUGEPAGE=0 uv run --no-sync python -m benchmarks.pdf_retrieval.prepare \
  --solution colmodernvbert-page-exact-maxsim --language en
MIR_DEVICE=cuda:0 uv run --no-sync python -m benchmarks.pdf_retrieval.run \
  --solution colmodernvbert-page-exact-maxsim --language en --cached-only

UV_PROJECT_ENVIRONMENT=.runtime/venvs/jina MIR_DEVICE=cuda:0 \
  uv run --no-sync python -m benchmarks.pdf_retrieval.prepare \
  --solution jina-v4-page-exact-maxsim
MIR_DEVICE=cuda:0 uv run --no-sync python -m benchmarks.pdf_retrieval.run \
  --solution jina-v4-page-exact-maxsim --cached-only
```

The preparation interface works for any chunked multi-vector Solution with a persistent cache. The explicit
`--cached-only` run option also supports other cached multi-vector Solutions. Missing entries fail without
loading a model or silently filling the cache. New visual caches use filenames derived from complete encoding
configuration fingerprints; image content determines page keys. Batch size and device ordinal do not change
cache identity. Query/document roles remain distinct. Smoke caches never share the benchmark runtime root.
Both new encoders normalize outputs in float32 before persisting them, matching exact scoring semantics;
BF16-only normalization can otherwise accumulate visible score drift across query tokens. Model release
collects reference cycles before clearing the CUDA allocator so scoring does not retain encoder weights.

Full scoring still requires a clean committed source tree for run provenance. The complete English pool has
5,702 pages and 500 queries; Chinese has 3,789 pages and 500 queries. Budget index memory from actual token counts
after preparation. English and Chinese pools are evaluated separately. A smoke peak does not validate all-page
encoding throughput or full-index memory usage.

Each retained page token requires 256 bytes in the float16 index and 512 bytes in the float32 cache before
compression. The index stays on the selected device; `block_vector_limit=1_000_000` limits scoring blocks,
so budget model weights and image activations during preparation, then block coalescing and query-score
matrices during cached scoring. A tiny smoke peak does not measure full-corpus memory. ColQwen token counts depend on page
resolution, and compressed cache size depends on the embeddings.

The 2026-10-23 cold smokes passed on RTX 3080 Ti with batch size 1. ColModernVBERT covered two English
pages/queries in 6.33 seconds; Jina covered two per language in 11.25 seconds. Both passed finite-value,
normalization, independent scoring, expected top-1, and fresh-instance cache-replay checks. Solution READMEs
record the measured memory and score errors. Jina encoding uses the pinned isolated environment;
its cache was also replayed successfully in the main environment with model loading forbidden.

For scheduling only, extrapolating the post-load smoke page times (0.57 s/page for ColModernVBERT and
0.46 s/page for Jina) gives about 55 minutes for English ColModernVBERT preparation and 75 minutes for
bilingual Jina preparation, including the corresponding query counts. Reserve roughly 1–2 hours and
1.5–3 hours respectively until actual PDF throughput is measured. At the smoke token counts, uncompressed
page caches would occupy about 2.40 GiB and 3.53 GiB; ColModernVBERT's English index would occupy 1.20 GiB,
and Jina's separate English/Chinese indexes 1.06/0.70 GiB, before allocator and scoring overhead.
Actual page shapes, token counts, query lengths, rendering, and storage latency can change these estimates.
The tiny smoke index does not provide a useful full exact-scoring time estimate; measure that phase after
preparation rather than treating smoke replay time as corpus throughput. No full preparation or scoring
run is part of this validation.
