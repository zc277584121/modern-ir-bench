# Modern IR Bench

Modern IR Bench is a code-native benchmark for modern information retrieval.
A Solution may be a model, a lexical method, a multi-stage retrieval pipeline,
or another system that implements a Task's runtime contract.

Python is the only configuration language. Task definitions, Dataset bindings,
Solution construction, and metrics are executable code; result evidence is kept
separately and the Hugging Face Space is generated from that evidence.

## Published benchmarks

### Ranked Retrieval v2.2

- 5,000 bilingual text documents
- 1,000 queries
- 6,575 positive binary Qrels
- 45 published lexical, dense, learned-sparse, reranking, and hybrid Solutions
- immutable historical Top-10 rankings in `results/modern-ir-v2.2/`
- Top-50 candidate evidence and reranker diagnostics in `results/modern-ir-v2.2/`
- frozen source tag: `ranked-retrieval-v2.2.0`
- Dataset: [`zc277584121/modern-ir-bench`](https://huggingface.co/datasets/zc277584121/modern-ir-bench)

Run every maintained Solution:

```bash
uv sync --group benchmark
uv run python -m benchmarks.ranked_retrieval.run
```

### PDF Retrieval v1

The PDF Task keeps the same semantic documents, queries, document IDs, and Qrels,
but replaces each text resource with a rendered PDF. Solutions decide how to read
pages and how to retrieve whole document IDs.

The 13 published Solutions cover OCR-based lexical, dense, and hybrid
retrieval plus OCR-free visual retrieval with Voyage Multimodal 3.5,
EmbeddingGemma 2, WeMM Embedding 2B, Gemini Embedding 2, Llama Nemotron Embed
VL 1B V2, and Ovis Omni Embedding 3B. None reads a native PDF text layer.
Published Top-10 rankings are in `results/modern-ir-pdf-v1/`.

The cross-modal catalog also includes an equal-weight RRF Solution that combines
full-document OCR/BM25 ranks with OCR-free page-image ranks.

The release also preserves two fixed-weight fusion runs as negative evidence.
Their diagnostics show that the text benchmark's 25/75 score weights do not
transfer reliably to heterogeneous PDF branches; RRF remains the recommended
default.

- Dataset: [`zc277584121/modern-ir-bench-pdf`](https://huggingface.co/datasets/zc277584121/modern-ir-bench-pdf)
- Design and Solution matrix: [`docs/pdf-solutions.md`](docs/pdf-solutions.md)

List the Solution choices:

```bash
uv run --group benchmark --group pdf python -m benchmarks.pdf_retrieval.run --help
```

Run the portable local dense Solution. The first run performs OCR on CPU; later
runs reuse its content-addressed cache:

```bash
MIR_DEVICE=cuda:0 MIR_MILVUS_LITE_PATH=.runtime/pdf-benchmark.db \
  uv run --group benchmark --group pdf --group pdf-cpu \
  python -m benchmarks.pdf_retrieval.run \
  --solution paddleocr-v5-mobile-bge-m3-full
```

GPU PaddleOCR uses the mutually exclusive `pdf-gpu` environment and is best run
first as the BM25 baseline to populate the shared OCR cache.

## Repository layout

```text
src/modern_ir_bench/             reusable framework and retrieval components
benchmarks/ranked_retrieval/     text Task, Dataset binding, and Solutions
benchmarks/pdf_retrieval/        PDF Task, Dataset binding, and Solution catalog
benchmarks/reporting.py          slices derived from saved rankings
benchmarks/publish.py            the only Space-data publisher
results/                         immutable result evidence
space/                           static Hugging Face Space and fusion explorer
docs/solutions/                  human-readable Solution notes
tests/                           framework and integration tests
```

There are four intentionally separate layers:

1. `src/modern_ir_bench/` defines reusable contracts and implementations.
2. `benchmarks/` composes concrete Tasks, Datasets, metrics, and Solutions.
3. `results/` preserves enough evidence to recompute published scores.
4. `space/` presents the generated leaderboard. Its optional Fusion lab recomputes
   exploratory rankings in the browser from exported atomic Top-50 evidence; it
   cannot alter the official result evidence.

## Core model

```text
Task
├── owns the evaluation protocol and metrics
├── binds one or more Hugging Face Dataset releases
└── runs every supplied Solution on every bound Dataset

Dataset
└── versioned data adapted to one Task contract

Solution
└── model, algorithm, retriever, pipeline, or agent under evaluation

Observation
└── per-sample evidence produced by Task × Dataset × Solution
```

`Task.run()` performs the Solution × Dataset Cartesian product. A Task defines the
data contract it needs; Dataset adapters preserve useful labels but do not dictate
batching, indexing, OCR, chunking, or retrieval policy. Those runtime choices belong
to Solutions.

## Publish the Space

The Space JSON is never hand-edited and no Task appends to an existing payload.
One command validates both result artifacts, recomputes metrics and slices, merges
the reports, and writes `space/data/results.json` from scratch:

```bash
uv run python -m benchmarks.publish
uv run python -m benchmarks.publish --check
```

Publishing requires a clean commit because Solution links are pinned to that commit.

The Fusion lab has a separate, deterministic export because it reads reusable
component rankings from the local candidate cache. It never calls a model, Milvus,
or an external API in the browser:

```bash
uv run python -m benchmarks.export_fusion_playground
```

Run the static Space locally:

```bash
cd space
python -m http.server 7861
```

## Add a Solution

Add the implementation and `SolutionSpec` under the matching Task's `solutions/`
package, then add its Markdown notes under `docs/solutions/`. The spec is the single
catalog entry for its id, title, description, tags, factory, and README path. The
Space derives the Solution name, filters, README preview, and source links from it.

Commit the source before running an evaluation so the result can pin an immutable
Git revision. Run only the new Solution:

```bash
uv run --group benchmark python -m benchmarks.ranked_retrieval.run \
  --solution <solution-id>

uv run --group benchmark --group pdf --group pdf-cpu \
  python -m benchmarks.pdf_retrieval.run \
  --solution <solution-id>
```

Set `MIR_MILVUS_LITE_PATH` on either command to use an isolated local database
instead of a Milvus Server. Visual models that depend on Qwen media utilities
add the `multimodal` dependency group.

Each command prints the verified run-artifact directory it created. Merge that
directory into the corresponding ranking release without rerunning existing
Solutions:

```bash
uv run python -m benchmarks.ranked_retrieval.release --run <run-artifact-directory>
uv run python -m benchmarks.pdf_retrieval.release --run <run-artifact-directory>
```

Commit the updated ranking evidence, then rebuild the Space payload with
`benchmarks.publish`. A Solution that has code but no published ranking is omitted
from the Space until its evidence is merged.

## Add a benchmark

Create a small package under `benchmarks/` containing a Task/Dataset definition,
one or more Solution constructors, and a direct `run.py` entry point. Reuse framework
components from `src/modern_ir_bench/`; add new abstractions only when two real Tasks
need them. No YAML registry or compatibility wrapper is required.

## Development

```bash
uv sync --group dev
uv run ruff check .
uv run pytest
```
