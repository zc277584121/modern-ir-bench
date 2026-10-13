# SPLADE v3 DistilBERT

This Solution pins the public SPLADE v3 DistilBERT checkpoint, a compact version of the canonical SPLADE v3 English learned-sparse retriever. Its native encoder limit is 512 tokens, so longer shared chunks are truncated by the model.

The benchmark deliberately runs the English checkpoint across the complete bilingual release. Language breakdowns therefore expose both its intended English behavior and its out-of-domain Chinese limitation instead of hiding the latter with language-specific routing.
