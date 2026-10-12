# Llama Nemotron Embed VL 1B V2 · page images

This OCR-free visual Solution renders each PDF page with the benchmark's shared
renderer and sends page images to
`nvidia/llama-nemotron-embed-vl-1b-v2:free` through OpenRouter. Text queries and
page images share one native 2048-dimensional space. Milvus performs exact
cosine search over pages before page hits are collapsed to unique document
rankings.

The Solution never reads native PDF text layers or OCR output. Embeddings are
cached by the exact rendered image content.
