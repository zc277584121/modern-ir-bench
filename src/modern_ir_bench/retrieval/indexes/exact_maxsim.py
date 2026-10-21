"""Exhaustive MaxSim search used as a multi-vector correctness reference."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from modern_ir_bench.embeddings.protocols import MultiVectors
from modern_ir_bench.retrieval.types import SearchHit


def _matrix(value: Any, *, dimension: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != dimension:
        raise ValueError(f"Expected a non-empty (*, {dimension}) token matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("multi-vector input contains a non-finite value")
    return matrix


@dataclass(frozen=True)
class TorchExactMaxSimIndex:
    """Score every document with MaxSim, without candidate generation."""

    device: str | None = None
    dtype: str = "float32"
    normalize: bool = True
    block_vector_limit: int | None = None

    def __post_init__(self) -> None:
        if self.block_vector_limit is not None and self.block_vector_limit < 1:
            raise ValueError("block_vector_limit must be positive")

    def open(self, *, dimension: int) -> TorchExactMaxSimSession:
        return TorchExactMaxSimSession(
            dimension=dimension,
            device=self.device,
            dtype=self.dtype,
            normalize=self.normalize,
            block_vector_limit=self.block_vector_limit,
        )


class TorchExactMaxSimSession:
    def __init__(
        self,
        *,
        dimension: int,
        device: str | None,
        dtype: str,
        normalize: bool,
        block_vector_limit: int | None,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        import torch

        dtypes = {"float16": torch.float16, "float32": torch.float32}
        if dtype not in dtypes:
            raise ValueError(f"Unsupported exact MaxSim dtype: {dtype!r}")
        self.dimension = dimension
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype_name = dtype
        self.dtype = dtypes[dtype]
        self.normalize = normalize
        self.block_vector_limit = block_vector_limit
        self._ids: list[str] = []
        self._id_set: set[str] = set()
        self._vector_blocks: list[Any] = []
        self._length_blocks: list[Any] = []
        self._sealed_vectors: list[Any] | None = None
        self._sealed_lengths: list[Any] | None = None
        self._vector_count = 0
        self._closed = False

    def add(self, ids: Sequence[str], vectors: MultiVectors) -> None:
        if self._closed:
            raise RuntimeError("Cannot add vectors to a closed exact MaxSim index")
        if self._sealed_vectors is not None:
            raise RuntimeError("Cannot add vectors after the index has been sealed")
        import torch
        from torch.nn import functional

        matrices = [_matrix(value, dimension=self.dimension) for value in vectors]
        normalized_ids = [str(item_id) for item_id in ids]
        if len(normalized_ids) != len(matrices):
            raise ValueError("ids and vectors must contain the same number of rows")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("ids must be unique within a batch")
        duplicates = self._id_set.intersection(normalized_ids)
        if duplicates:
            raise ValueError(f"Duplicate ids across batches: {sorted(duplicates)}")
        if not matrices:
            return

        block = torch.from_numpy(np.concatenate(matrices, axis=0)).to(self.device)
        if self.normalize:
            block = functional.normalize(block, dim=1)
        block = block.to(self.dtype)
        lengths = torch.tensor([matrix.shape[0] for matrix in matrices], dtype=torch.int64, device=self.device)
        self._ids.extend(normalized_ids)
        self._id_set.update(normalized_ids)
        self._vector_blocks.append(block)
        self._length_blocks.append(lengths)
        self._vector_count += block.shape[0]

    def seal(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot seal a closed exact MaxSim index")
        if self._sealed_vectors is not None:
            return
        if not self._vector_blocks:
            raise ValueError("Cannot seal an empty exact MaxSim index")
        self._sealed_vectors, self._sealed_lengths = self._coalesce_blocks()

    def _coalesce_blocks(self) -> tuple[list[Any], list[Any]]:
        import torch

        if self.block_vector_limit is None:
            vectors = [torch.cat(self._vector_blocks, dim=0)]
            lengths = [torch.cat(self._length_blocks, dim=0)]
            self._vector_blocks.clear()
            self._length_blocks.clear()
            return vectors, lengths

        pending_vectors = deque(self._vector_blocks)
        pending_lengths = deque(self._length_blocks)
        self._vector_blocks.clear()
        self._length_blocks.clear()
        vectors: list[Any] = []
        lengths: list[Any] = []
        while pending_vectors:
            vector_group = []
            length_group = []
            vector_count = 0
            while pending_vectors and (not vector_group or vector_count < self.block_vector_limit):
                block = pending_vectors[0]
                if vector_group and vector_count + block.shape[0] > self.block_vector_limit:
                    break
                vector_group.append(pending_vectors.popleft())
                length_group.append(pending_lengths.popleft())
                vector_count += block.shape[0]
            vectors.append(torch.cat(vector_group, dim=0))
            lengths.append(torch.cat(length_group, dim=0))
        return vectors, lengths

    def search(self, query_vectors: MultiVectors, *, top_k: int) -> list[list[SearchHit]]:
        if self._closed or self._sealed_vectors is None or self._sealed_lengths is None:
            raise RuntimeError("The exact MaxSim index must be sealed before search")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        import torch
        from torch.nn import functional

        limit = min(top_k, len(self._ids))
        output: list[list[SearchHit]] = []
        with torch.inference_mode():
            for value in query_vectors:
                query = torch.from_numpy(_matrix(value, dimension=self.dimension)).to(self.device)
                if self.normalize:
                    query = functional.normalize(query, dim=1)
                query = query.to(self.dtype)
                score_blocks = []
                for vectors, block_lengths in zip(
                    self._sealed_vectors,
                    self._sealed_lengths,
                    strict=True,
                ):
                    token_scores = query @ vectors.T
                    lengths = block_lengths.unsqueeze(0).expand(query.shape[0], -1)
                    score_blocks.append(
                        torch.segment_reduce(
                            token_scores,
                            "max",
                            lengths=lengths,
                            axis=1,
                        ).float().sum(dim=0)
                    )
                document_scores = torch.cat(score_blocks)
                scores = document_scores.float().cpu().tolist()
                ranked = sorted(
                    zip(self._ids, scores, strict=True),
                    key=lambda item: (-item[1], item[0]),
                )[:limit]
                output.append([SearchHit(id=item_id, score=float(score)) for item_id, score in ranked])
        return output

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "torch-exact-maxsim",
            "device": str(self.device),
            "dtype": self.dtype_name,
            "dimension": self.dimension,
            "count": len(self._ids),
            "token_vectors": self._vector_count,
            "blocks": len(self._sealed_vectors or ()),
            "normalize": self.normalize,
            "exact": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._ids.clear()
        self._id_set.clear()
        self._vector_blocks.clear()
        self._length_blocks.clear()
        self._sealed_vectors = None
        self._sealed_lengths = None
        self._closed = True
        if self.device.type == "cuda":
            import torch

            torch.cuda.empty_cache()
