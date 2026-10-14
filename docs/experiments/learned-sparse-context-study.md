# Learned-sparse context-window study

This study isolates the English half of Modern IR Ranked Retrieval v2.2: 2,500 documents, 500 queries, and 17,170 binary Qrels. All values below are NDCG@10 over the same data and retrieval implementation.

## Why the previous configuration was invalid

The normal benchmark chunker targets 768 tokens using the pinned BGE-M3 boundary tokenizer. Several learned-sparse checkpoints accept at most 512 of their own tokens. On the English corpus, 2,627 of 4,532 normal chunks (57.97%) exceed 512 SPLADE tokens and were silently truncated during encoding.

The tested short-context policy targets 384 BGE-M3 tokens, permits 256–448, and overlaps 64. It produces 8,156 English chunks; none exceed 512 SPLADE tokens.

## Controlled results

| Model | Normal chunks | Short-context chunks | Delta |
| --- | ---: | ---: | ---: |
| SPLADE v1 `flops_best` (2021, original sum pooling) | 0.6768 | 0.6847 | +0.0079 |
| SPLADE v2 max | 0.6854 | 0.7016 | +0.0162 |
| SPLADE-v3 DistilBERT | 0.7205 | 0.7285 | +0.0081 |
| SPLADE-v3 full | 0.7342 | 0.7450 | +0.0108 |
| OpenSearch doc-v2 distill | 0.7230 | 0.7345 | +0.0115 |
| OpenSearch multilingual v1 | 0.6934 | 0.7176 | +0.0242 |
| Granite Embedding 30M Sparse, official pruning | 0.7391 | 0.7493 | +0.0102 |

The historical SPLADE v1 checkpoint is the official repository's `flops_best` artifact at commit `a8af33f17901c8bf30fdeb386b82e4ac4feeaaae` (SHA256 `2c2d07a3edac1dd0bc14f26b1b7e1b95d2b7c0902b0c24c8268cd37b1aad0efd`). The full SPLADE-v3 measurement uses Castorini's full-precision ONNX conversion at revision `51342029dd50ef0845c16081bd423039e7f61717`; it is not the smaller DistilBERT checkpoint.

Granite's official pruning retains the 50 largest query dimensions and 192 largest document dimensions, with Milvus build/search drop ratios of 0.2. On normal chunks it raises Granite from 0.7168 to 0.7391; using the short-context policy raises it again to 0.7493.

## Rejected alternatives

- Changing every Solution to short chunks is not valid. BM25 falls from 0.7127 to 0.6519 because this changes the lexical retrieval unit rather than fixing model truncation.
- Keeping normal chunks and merging several model-sized window vectors by coordinate-wise maximum also underperforms: Granite 0.7325, OpenSearch doc-v2 0.7086, and SPLADE-v3 DistilBERT 0.7045. The merge accumulates expansion noise into one sparse vector.
- Selecting an arbitrary chunk size for every individual checkpoint would make the catalog difficult to compare and maintain.

## Adopted rule

Chunking is part of a Solution, not a Dataset invariant. Models with a 512-token document limit use one shared short-context policy. Models that can consume the normal chunks, and lexical baselines such as BM25, retain the normal policy. Model-card-specific sparse pruning remains an explicit embedding/index setting rather than being hidden in the chunker.
