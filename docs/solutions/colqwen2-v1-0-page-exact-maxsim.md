# ColQwen2 v1.0 · page exact MaxSim

This OCR-free Solution renders PDF pages as images, encodes them with `vidore/colqwen2-v1.0`, and exhaustively
scores every page with Exact MaxSim. Queries and pages contain multiple 128-dimensional vectors; each PDF
receives its best page score.

The [official model configuration](https://huggingface.co/vidore/colqwen2-v1.0) is loaded through the shared
`SentenceTransformersMultiVectorEmbedding` component using Sentence Transformers `MultiVectorEncoder`.
The pinned revision is `2b6ac8fb37f46a49e4841e599583d00ae8a20117`. Its adapter loads the
`vidore/colqwen2-base` backbone. The published configuration supplies the training-time query prefix,
dynamic image processing, projection, token mask, and adapter weight mapping. All tokens retained by
the official mask participate in MaxSim.

The shared PDF renderer uses scale 2, a maximum side of 1600 pixels, and JPEG quality 88. Model inference
uses bfloat16 on supported CUDA devices. The Exact MaxSim index stores normalized vectors in float16;
the SQLite embedding cache stores compressed float32 matrices in `.runtime/multi-vector/colqwen2-v1.0-pages.sqlite3`.
Set `MIR_DEVICE` to select a device and `MIR_COLQWEN2_V1_0_BATCH_SIZE` to override the default batch size of 1.

See the [visual Exact MaxSim run instructions](../pdf-solutions.md#visual-exact-maxsim) for the isolated smoke
test and full benchmark commands. GPU memory grows with the total page-token count, in addition to model
weights and inference activations.
