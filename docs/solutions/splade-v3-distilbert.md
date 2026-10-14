# SPLADE v3 DistilBERT

This Solution pins the public SPLADE v3 DistilBERT checkpoint, a compact version of the canonical SPLADE v3 English learned-sparse retriever. Documents use the shared short-context policy (384 target tokens, 448 maximum, 64 overlap), so the checkpoint's 512-token limit does not silently truncate content.

The benchmark deliberately runs the English checkpoint across the complete bilingual release. Language breakdowns therefore expose both its intended English behavior and its out-of-domain Chinese limitation instead of hiding the latter with language-specific routing.
