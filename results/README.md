# Result evidence

Each subdirectory is one immutable, published evaluation artifact. These files are
neither benchmark Datasets nor Solution implementations.

- `modern-ir-v2.2/` contains historical Top-10 rankings for seven text-retrieval
  Solutions. Metrics are recomputed from those rankings and the pinned public Dataset.
- `modern-ir-pdf-v1/` contains query-level observations from the first live PDF
  baseline. Metrics are recomputed from those observations.

Every artifact has a manifest with its Dataset revision, inventory, and checksums.
`python -m benchmarks.publish` validates the manifests and rebuilds the Space payload
from scratch. Generated run directories belong under ignored `artifacts/`; only
reviewed evidence is promoted here.
