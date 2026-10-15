# SPLADE v3 Full

This Solution pins the full BERT-base SPLADE v3 checkpoint. It is the stronger,
larger counterpart to the published DistilBERT variant and uses the same English
WordPiece vocabulary for learned query and document expansion.

Documents use the shared short-context policy: 384 target tokens, 448 maximum,
and 64 tokens of overlap. This keeps every chunk below the model's 512-token
limit instead of silently truncating the benchmark's normal chunks. Milvus stores
the resulting sparse vectors in a sparse inverted index and scores them with inner
product before chunk hits are collapsed to document rankings.

The checkpoint is English-only. The benchmark keeps that limitation visible in
the language slices rather than adding an implicit fallback retriever. Its model
card declares the `CC-BY-NC-SA-4.0` license; this repository references but does
not redistribute the checkpoint.
