# EmbeddingGemma 2 · page images

This OCR-free visual Solution pins `google/embeddinggemma-2` to revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`. It loads the text and vision
encoders, renders every PDF page with the benchmark's shared renderer, embeds
page images, and embeds queries with the model's `SearchQuery` prompt route.
Milvus performs exact cosine search over 768-dimensional vectors, after which
page hits are collapsed to whole-document rankings.

Inference uses bfloat16 on supported GPUs and float32 otherwise. It never reads
the native PDF text layer and performs no OCR or reranking.
