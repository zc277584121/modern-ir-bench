# Modern IR Bench

Modern IR Bench is a code-native benchmark for modern information retrieval solutions. A participant can be a model, an algorithm, a multi-stage retrieval pipeline, a RAG system, or an agent memory system.

The project is intentionally Python-only: executable benchmark modules are the source of truth, and every published score links back to an immutable Git commit and the exact code that produced it.

## Current release

The current reviewed release is Modern IR Ranked Retrieval v2.2: a bilingual synthetic
retrieval benchmark with 5,000 documents, 1,000 queries, and 34,756 binary Qrels,
including 6,575 relevant pairs.
Its public leaderboard compares seven retrieval Solutions spanning full-document and
chunked BM25, local dense and learned-sparse models, and hosted dense embeddings.

v2.2 is a schema-only migration from v2.1. It renames the Query field `task` to
`query_intent`; Query IDs and text, Corpus documents, Qrels, rankings, and scores are unchanged.

The earlier deterministic mock showcase remains available as a framework smoke test:

```bash
uv sync
uv run python -m benchmarks.mock_showcase
uv run pytest
```

The reviewed retrieval release can be replayed through the same public
`Task` / `Solution` / `Metric` path. The Dataset contains only corpus, queries,
and binary Qrels. The executable module pins the immutable
[`zc277584121/modern-ir-bench`](https://huggingface.co/datasets/zc277584121/modern-ir-bench)
Dataset revision and recomputes every leaderboard metric from the versioned run artifact
under `results/modern-ir-v2.2/`:

```bash
uv run python -m benchmarks.modern_ir
```

This validates all seven saved retrieval routes against the expanded Qrels and writes
framework observations, long-form results, and a local Space payload under
`artifacts/modern-ir-retrieval/`. The replay is a reproducibility check; model
inference and indexing remain ordinary Solution implementations.

Run a fresh BGE-M3 inference/index/search path against independent English and Chinese
candidate pools:

```bash
uv run --group benchmark python -m benchmarks.modern_ir_bge_m3_live
```

This uses the public `TokenTextChunker`, `ChunkedDenseRetrievalSolution`,
`SentenceTransformersEmbedding`, and `MilvusDenseIndex` components. It compares every
fresh Top-10 ranking with the immutable saved ranking and writes the audit under
`artifacts/modern-ir-bge-m3-live/`. Set `MIR_DEVICE`, `MIR_BATCH_SIZE`, and
`MIR_MILVUS_URI` when the local defaults are not appropriate.

Run the Space locally:

```bash
cd space
python -m http.server 7861
```

## Core model

```text
Task
├── declares its Hugging Face Dataset contract
├── owns metrics and evaluation semantics
└── decides how to call a Solution

Solution
└── model, algorithm, retriever, pipeline, or agent being evaluated

Observation
└── sample-level evidence produced by a Task

Metric
└── scores task-specific Observations
```

Datasets use Hugging Face `Dataset` and `IterableDataset` directly. Adapters are ordinary Python functions, and ingestion/search phases remain explicit inside each task and solution.

## Retrieval indexes

Milvus is the default production-facing dense index path. The three deployment
families are explicit so unsupported settings fail early instead of being silently
ignored:

```python
from modern_ir_bench.retrieval.indexes import NumpyFlatIndex, VerifiedDenseIndex
from modern_ir_bench.retrieval.indexes.milvus import MilvusDenseIndex, MilvusLite

index = VerifiedDenseIndex(
    primary=MilvusDenseIndex(
        target=MilvusLite("artifacts/example.db"),
        metric="COSINE",
    ),
    oracle=NumpyFlatIndex(metric="COSINE"),
)
```

The primary Milvus result is returned to the Task. The independent exhaustive
NumPy search retains index-recall evidence that helps separate model quality from
index, consistency, or integration errors. Both backends receive the exact same
embedding batches through one `DenseIndex` lifecycle: `open`, `add`, `seal`,
`search`, and `close`.

Milvus sessions always use an isolated temporary collection, Strong consistency,
and record the requested and actual index descriptions plus client/server versions.
The framework currently validates Milvus Lite with a real integration test; Server
and Zilliz Cloud use the same client implementation with deployment-specific index
validation.

## Add an experiment

Create an executable module under `benchmarks/`, construct datasets, tasks, metrics, and solutions, then run it directly:

```bash
uv run python -m benchmarks.your_experiment
```

No YAML registry or parallel configuration language is required.

## Publishing model

The Space is a read-only view of maintainer-approved results. Contributions arrive through GitHub pull requests; there is no public model upload or shared evaluation runtime. Dataset releases are immutable, and new data is published as a new version.

The full refactor design is maintained in `.local.PLAN.md`. Earlier experimental
implementations and generated artifacts remain recoverable from Git history and the
archived experiment branches; they are intentionally excluded from the current tree.
