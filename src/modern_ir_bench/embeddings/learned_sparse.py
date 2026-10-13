"""Lazy local learned-sparse embedding implementations."""

from __future__ import annotations

from collections.abc import Sequence
from types import MethodType
from typing import Any, Literal

from modern_ir_bench.embeddings.sparse import SparseVector


class SentenceTransformersSparseEmbedding:
    """Adapt a pinned Sentence Transformers SparseEncoder to the benchmark protocol."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        if not model or not revision:
            raise ValueError("model and revision must be non-empty")
        if batch_size < 1 or (max_length is not None and max_length < 1):
            raise ValueError("batch size and max length must be positive")
        self.model_id = model
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from sentence_transformers import SparseEncoder

        self._model = SparseEncoder(
            self.model_id,
            revision=self.revision,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        if self.max_length is not None:
            self._model.max_seq_length = self.max_length
        return self._model

    @staticmethod
    def _to_vectors(tensor: Any) -> list[SparseVector]:
        if not tensor.is_sparse:
            tensor = tensor.to_sparse()
        tensor = tensor.coalesce().detach().cpu()
        rows, columns = tensor.indices().tolist()
        values = tensor.values().tolist()
        result: list[SparseVector] = [{} for _ in range(tensor.shape[0])]
        for row, column, value in zip(rows, columns, values, strict=True):
            if value > 0:
                result[row][int(column)] = float(value)
        return result

    def _encode(self, inputs: Sequence[Any], *, kind: Literal["document", "query"]) -> list[SparseVector]:
        texts = [str(value) for value in inputs]
        if not texts:
            return []
        model = self._load_model()
        encode = model.encode_document if kind == "document" else model.encode_query
        output = encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        return self._to_vectors(output)

    def encode_sparse_documents(self, inputs: Sequence[Any]) -> list[SparseVector]:
        return self._encode(inputs, kind="document")

    def encode_sparse_queries(self, inputs: Sequence[Any]) -> list[SparseVector]:
        return self._encode(inputs, kind="query")

    def release(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class GteMultilingualSparseEmbedding:
    """Generate GTE multilingual token-weight vectors without materializing dense vocabulary logits."""

    def __init__(
        self,
        *,
        revision: str,
        model: str = "Alibaba-NLP/gte-multilingual-base",
        batch_size: int = 16,
        device: str | None = None,
        max_length: int = 1024,
        use_fp16: bool = True,
    ) -> None:
        if not model or not revision:
            raise ValueError("model and revision must be non-empty")
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch size and max length must be positive")
        self.model_id = model
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.use_fp16 and device.startswith("cuda") else None
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self._model = AutoModelForTokenClassification.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device)
        self._repair_legacy_runtime(self._model, torch)
        self._model.eval()
        return self._model, self._tokenizer

    @staticmethod
    def _repair_legacy_runtime(model: Any, torch: Any) -> None:
        """Restore non-persistent buffers and one Transformers 4 helper used by this pinned checkpoint."""
        backbone = model.new
        embeddings = backbone.embeddings
        config = model.config
        embeddings.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings, device=model.device),
            persistent=False,
        )

        rotary = embeddings.rotary_emb
        inverse_frequency = 1.0 / (
            rotary.base
            ** (torch.arange(0, rotary.dim, 2, device=model.device).float() / rotary.dim)
        )
        rotary.register_buffer("inv_freq", inverse_frequency, persistent=False)
        cache_length = int(config.max_position_embeddings * config.rope_scaling["factor"])
        rotary._set_cos_sin_cache(cache_length, model.device, next(model.parameters()).dtype)

        if not hasattr(backbone, "get_extended_attention_mask"):

            def get_extended_attention_mask(instance: Any, attention_mask: Any, _: Any) -> Any:
                dtype = next(instance.parameters()).dtype
                visible = attention_mask[:, None, None, :].to(dtype=dtype)
                return (1.0 - visible) * torch.finfo(dtype).min

            backbone.get_extended_attention_mask = MethodType(get_extended_attention_mask, backbone)

    def _encode(self, inputs: Sequence[Any]) -> list[SparseVector]:
        import torch

        texts = [str(value) for value in inputs]
        if not texts:
            return []
        model, tokenizer = self._load_model()
        special_ids = set(tokenizer.all_special_ids)
        result: list[SparseVector] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(model.device) for key, value in batch.items()}
                weights = torch.relu(model(**batch, return_dict=True).logits).squeeze(-1)
                for token_ids, token_weights, attention_mask in zip(
                    batch["input_ids"].cpu().tolist(),
                    weights.float().cpu().tolist(),
                    batch["attention_mask"].cpu().tolist(),
                    strict=True,
                ):
                    vector: SparseVector = {}
                    for token_id, weight, active in zip(
                        token_ids,
                        token_weights,
                        attention_mask,
                        strict=True,
                    ):
                        if active and token_id not in special_ids and weight > 0:
                            vector[int(token_id)] = max(vector.get(int(token_id), 0.0), float(weight))
                    result.append(vector)
        return result

    def encode_sparse_documents(self, inputs: Sequence[Any]) -> list[SparseVector]:
        return self._encode(inputs)

    def encode_sparse_queries(self, inputs: Sequence[Any]) -> list[SparseVector]:
        return self._encode(inputs)

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
