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
- seven lexical, dense, and learned-sparse Solutions
- immutable historical Top-10 rankings in `results/modern-ir-v2.2/`
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

The first baseline OCRs every page with PP-OCRv5 Mobile and then runs full-document
Milvus BM25. It never reads a native PDF text layer. Its preserved live observations
are in `results/modern-ir-pdf-v1/`.

- Dataset: [`zc277584121/modern-ir-bench-pdf`](https://huggingface.co/datasets/zc277584121/modern-ir-bench-pdf)

Run the baseline:

```bash
uv sync --group pdf --group pdf-gpu
MIR_OCR_DEVICE=gpu:0 uv run python -m benchmarks.pdf_retrieval.run
```

Use `pdf-cpu` and `MIR_OCR_DEVICE=cpu` for a portable CPU run.

## Repository layout

```text
src/modern_ir_bench/             reusable framework and retrieval components
benchmarks/ranked_retrieval/     text Task, Dataset binding, and Solutions
benchmarks/pdf_retrieval/        PDF Task, Dataset binding, and OCR baseline
benchmarks/reporting.py          slices derived from saved rankings
benchmarks/publish.py            the only Space-data publisher
results/                         immutable result evidence
space/                           static, read-only Hugging Face Space
docs/solutions/                  human-readable Solution notes
tests/                           framework and integration tests
```

There are four intentionally separate layers:

1. `src/modern_ir_bench/` defines reusable contracts and implementations.
2. `benchmarks/` composes concrete Tasks, Datasets, metrics, and Solutions.
3. `results/` preserves enough evidence to recompute published scores.
4. `space/` presents a generated leaderboard and contains no evaluation logic.

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

Run the static Space locally:

```bash
cd space
python -m http.server 7861
```

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
