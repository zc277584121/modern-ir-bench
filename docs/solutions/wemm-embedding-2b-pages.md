# WeMM Embedding 2B · page images

This OCR-free visual Solution pins `tencent/WeMM-Embedding-2B` to revision
`bbd6cd4bf52cfc6716f752a2df80b2706720bd95`. It renders every PDF page with the
shared renderer and uses WeMM's distinct document and query encoding routes.
Milvus performs exact cosine search over 2048-dimensional vectors, after which
page hits are collapsed to whole-document rankings.

Inference uses bfloat16 on supported GPUs and float32 otherwise. It never reads
the native PDF text layer and performs no OCR or reranking.
