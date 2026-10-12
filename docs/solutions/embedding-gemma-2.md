# EmbeddingGemma 2

This lightweight local dense Solution pins `google/embeddinggemma-2` to revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`. Only its text encoder is loaded.
Queries and documents use the model's `SearchQuery` and `Document` prompt routes,
then Milvus performs exact cosine search over 768-dimensional vectors generated
from the benchmark-wide shared chunks.

Inference uses bfloat16 on supported GPUs and float32 otherwise, following the
model card's warning that float16 can produce unstable embeddings.
