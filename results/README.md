# Result evidence

Each subdirectory is one immutable, published evaluation artifact. These files are
neither benchmark Datasets nor Solution implementations.

- `modern-ir-v2.2/` contains complete bilingual Top-10 rankings for 43 text-retrieval Solutions,
  Top-50 evidence for the two reranked candidate paths, and a diagnostic report.
- `modern-ir-pdf-v1/` contains Top-10 rankings for 13 PDF-retrieval Solutions and
  fusion and reranker diagnostics.
- `candidate-pools/` compares the coverage, overlap, and oracle ceiling of every
  Top-50 route exported for the Fusion lab.
- `modern-ir-v2.2/slices/en/` contains the three published English-only text Solutions.
- `modern-ir-pdf-v1/slices/en/` contains ColBERTv2, ColSmol 256M/500M, and ColPali v1.3.

Metrics and slices are recomputed from these rankings and their pinned public
Datasets. The diagnostic JSON files preserve candidate coverage, reranker deltas,
and fixed-score-fusion comparisons without changing Qrels.

Every artifact has a manifest with its Dataset revision, inventory, and checksums.
`python -m benchmarks.publish` validates the manifests and rebuilds the Space payload
from scratch. Generated run directories belong under ignored `artifacts/`; only
reviewed evidence is promoted here.

## Language coverage

Every entry in both Solution catalogs must explicitly declare supported languages;
`SolutionSpec` has no language default. A complete bilingual release requires
every Solution to support English and Chinese and cover all 1,000 queries. A language
release requires all 500 queries and the same 2,500-document candidate pool for that language.
English-only results contribute to the English query breakdown. They have no overall,
Chinese, or mixed-language breakdown metric records. The Space payload retains their
metadata with `solutions[].languages`, and provides each scope's `required_languages`
and `excluded_solutions` with reasons. Dataset-level `coverage` links each Solution to
its evidence, including evaluated languages, query counts, checksums, and provenance.
Supported languages are declared once in Solution metadata; evaluated languages come
from evidence rather than an independently maintained summary.

Run or merge one language with `--language en` (or `zh`). Default full runs select only
Solutions covering both languages; explicitly selecting an unsupported Solution fails.

```bash
uv run python -m benchmarks.ranked_retrieval.run --language en --solution granite-30m-sparse-chunk
uv run python -m benchmarks.pdf_retrieval.run --language en --solution colsmol-500m-page-exact-maxsim
```

Derive a language ranking release from an existing complete run or ranking release:

```bash
uv run python -m benchmarks.pdf_retrieval.release --language en --derive-from /path/to/complete-run
uv run python -m benchmarks.ranked_retrieval.release --language en \
  --derive-from /path/to/complete-ranking-release --solution granite-30m-sparse-chunk
```

`--derive-from` and `--solution` can be repeated. Derivation verifies source checksums,
Dataset revision, exact query coverage, candidate-pool membership, ranking depth, and
run aggregates against the pinned Qrels. It selects existing query rankings without
retrieval or padding. The derived manifest preserves original source provenance and
the source artifact's manifest/data checksums. Original run files remain unchanged.
Use `--run /path/to/english-run` to merge an already scoped run. Both release commands
default to `slices/<language>/` inside their result directory when a language is selected.

Rebuild the cross-task candidate report from stored Fusion-lab evidence without
calling a model, Milvus, or an external API:

```bash
uv run python -m benchmarks.candidate_pools
```
