# Voyage Multimodal 3.5 · Page Images

This Solution retrieves PDF pages visually and never performs OCR.

## Pipeline

1. Render every PDF page at 2x scale.
2. Resize pages to a maximum side of 1600 pixels and cache them as JPEG images.
3. Encode every page with `voyage-multimodal-3.5` at 1024 dimensions.
4. Encode the text query in the same multimodal embedding space.
5. Search page vectors with exact cosine similarity in Milvus.
6. Keep each document's best page score and return document-level rankings.

Page-level indexing avoids forcing multi-page PDFs into one image or one dense
vector. The page renderer and embeddings are content-addressed, so interrupted
runs resume without changing rankings.
