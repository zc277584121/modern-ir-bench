"""Dataset loaders that adapt published artifacts to task-owned contracts."""

from modern_ir_bench.datasets.pdf_release import (
    load_pdf_retrieval_hub,
    load_pdf_retrieval_release,
)
from modern_ir_bench.datasets.retrieval_release import (
    load_ranked_retrieval_hub,
    load_ranked_retrieval_release,
)

__all__ = [
    "load_pdf_retrieval_hub",
    "load_pdf_retrieval_release",
    "load_ranked_retrieval_hub",
    "load_ranked_retrieval_release",
]
