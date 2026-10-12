# PDF Retrieval Solution Plan

The PDF Task always evaluates document identifiers. A Solution may read a PDF in
any way, but page or chunk hits must be collapsed back to unique document IDs
before scoring.

## Maintained catalog

| Solution | PDF reading | Retrieval unit | Representation | Purpose |
| --- | --- | --- | --- | --- |
| PP-OCRv5 Mobile + BM25 | OCR | whole document | lexical | portable baseline |
| PP-OCRv5 Mobile + BGE-M3 | OCR | whole document | local dense | isolate dense versus lexical retrieval |
| PP-OCRv5 Mobile + BM25/BGE-M3 RRF | OCR | whole document | local hybrid | combine exact terms and semantic matches |
| PP-OCRv5 Mobile + Voyage 4 Large | OCR | whole document | hosted dense | compare a stronger hosted text model |
| Voyage Multimodal 3.5 | rendered images | page | visual dense | test OCR-free retrieval |
| PP-OCRv5 Mobile BM25 + Voyage Multimodal 3.5 RRF | OCR + rendered images | document + page | lexical + visual dense | combine exact text and visual evidence |

All dense indexes use exact cosine search through Milvus FLAT. The visual
Solution keeps the best page score for each document. This makes index behavior
an invariant while the document representation changes.

## Next controlled additions

1. **PP-OCRv5 Server + BM25/BGE-M3** changes only OCR quality. It should be added
   after the current OCR and dense baselines have full results because it requires
   a second complete OCR pass.
2. **ColQwen visual late interaction** is the preferred open-source visual
   baseline. It requires a Sentence Transformers v6 migration and a true
   multi-vector scorer, so it belongs in a separate implementation step rather
   than being approximated by a single-vector index.

Native PDF text extraction is deliberately excluded: image-only and damaged-text
PDFs would silently take a different path, which would make it a format shortcut
rather than a valid baseline for this Dataset.

## Runtime and caches

- OCR text is cached by PDF content hash.
- Rendered pages are cached by renderer configuration and PDF content hash.
- Hosted embeddings are cached by model, mode, dimension, and input content.
- Voyage Multimodal uses batches of 64 pages by default. With the renderer's
  1,600-pixel maximum side, even 64 square pages remain below the API's 320K
  multimodal-token request limit. Override it with
  `MIR_VOYAGE_MULTIMODAL_BATCH_SIZE` when project limits differ.
- Two multimodal requests may run concurrently so network and service latency do
  not serialize the full indexing pass. Set `MIR_VOYAGE_MULTIMODAL_CONCURRENCY`
  to match the active project RPM limit.
- Every benchmark run creates isolated Milvus collections and drops them after
  the evaluation session closes.

The current Voyage organization has free capacity but reduced API rate limits.
The local BGE-M3 Solution is therefore the first choice for an immediate full
run; the hosted visual Solution is resumable but rate-limit bound.
