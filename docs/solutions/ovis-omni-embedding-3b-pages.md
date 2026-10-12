# Ovis Omni Embedding 3B · page images

This self-hosted visual Solution uses the pinned Apache-2.0
`ATH-MaaS/Ovis-Omni-Embedding-3B` checkpoint. It follows the model's native
Qwen2.5-Omni processor and chat template, removes the unused language-modeling
head and audio tower after loading, takes the last non-padding hidden state,
and L2-normalizes the resulting native 2048-dimensional vector.

Queries use a visual-document retrieval instruction. PDF pages are candidate
images under the model's generic representation instruction. Milvus performs
exact cosine search over pages before page hits are collapsed to unique PDF
document rankings. The Solution never reads native PDF text layers or OCR
output, and embeddings are cached by exact rendered image content.
