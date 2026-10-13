# OCR Voyage 4 Large + Voyage Multimodal 3.5 · RRF

This hybrid combines two complementary views of each PDF through Voyage's
official API:

- PP-OCRv5 Mobile reads every page, and Voyage 4 Large retrieves complete OCR
  documents from their text.
- Voyage Multimodal 3.5 embeds rendered PDF pages directly, then page hits are
  collapsed to whole-document rankings.

Each route contributes its Top 50 documents. Equal-weight reciprocal rank
fusion combines the rankings with `k=60`, and the benchmark evaluates the final
Top 10. Stored route evidence also supports offline fusion experiments without
repeating OCR, embedding, or retrieval.
