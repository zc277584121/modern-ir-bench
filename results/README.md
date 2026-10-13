# Result evidence

Each subdirectory is one immutable, published evaluation artifact. These files are
neither benchmark Datasets nor Solution implementations.

- `modern-ir-v2.2/` contains Top-10 rankings for 21 text-retrieval Solutions,
  Top-50 evidence for the two reranked candidate paths, and a diagnostic report.
- `modern-ir-pdf-v1/` contains Top-10 rankings for 13 PDF-retrieval Solutions and
  a fusion diagnostic report.

Metrics and slices are recomputed from these rankings and their pinned public
Datasets. The diagnostic JSON files preserve candidate coverage, reranker deltas,
and fixed-score-fusion comparisons without changing Qrels.

Every artifact has a manifest with its Dataset revision, inventory, and checksums.
`python -m benchmarks.publish` validates the manifests and rebuilds the Space payload
from scratch. Generated run directories belong under ignored `artifacts/`; only
reviewed evidence is promoted here.
