"""Dataset loaders that adapt published artifacts to task-owned contracts."""

from modern_ir_bench.datasets.retrieval_release import (
    load_ranked_retrieval_hub,
    load_ranked_retrieval_release,
)

__all__ = ["load_ranked_retrieval_hub", "load_ranked_retrieval_release"]
