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

All dense indexes use exact cosine search through Milvus FLAT. The visual
Solution keeps the best page score for each document. This makes index behavior
an invariant while the document representation changes.

## Next controlled additions

1. **PP-OCRv5 Server + BM25/BGE-M3** changes only OCR quality. It should be added
   after the current OCR and dense baselines have full results because it requires
   a second complete OCR pass.
2. **Visual/text fusion** combines the saved OCR and visual rankings after both
   standalone paths have complete results. The maintained BM25/BGE-M3 RRF
   Solution already measures runtime hybrid retrieval rather than post-hoc score
   manipulation.
3. **ColQwen visual late interaction** is the preferred open-source visual
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
- Every benchmark run creates isolated Milvus collections and drops them after
  the evaluation session closes.

The current Voyage organization has free capacity but reduced API rate limits.
The local BGE-M3 Solution is therefore the first choice for an immediate full
run; the hosted visual Solution is resumable but rate-limit bound.
