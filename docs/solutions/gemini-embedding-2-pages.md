# Gemini Embedding 2 · page images

This OCR-free visual Solution renders each PDF page with the benchmark's shared
renderer and sends page images to `google/gemini-embedding-2` through OpenRouter.
Text queries and page images share one native 3072-dimensional space. Milvus
performs exact cosine search over pages before page hits are collapsed to unique
document rankings.

The Solution never reads native PDF text layers or OCR output. Embeddings are
cached by the exact rendered image content.
