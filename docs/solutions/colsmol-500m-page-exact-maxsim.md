# ColSmol 500M · page exact MaxSim

This OCR-free Solution renders PDF pages as images, encodes them with `vidore/colSmol-500M`, and exhaustively
scores every page with Exact MaxSim. Queries and pages contain multiple 128-dimensional vectors; each PDF
receives its best page score.

The [official model configuration](https://huggingface.co/vidore/colSmol-500M) is loaded through the shared
`SentenceTransformersMultiVectorEmbedding` component using Sentence Transformers `MultiVectorEncoder`.
The pinned revision is `0aaa9726104ce485884c7b8faa8a58a72d5fdbe7`. Its adapter loads the
`vidore/ColSmolVLM-Instruct-500M-base` backbone. The published configuration supplies the training-time
prompts, projection, token mask, and adapter weight mapping.

The shared PDF renderer uses scale 2, a maximum side of 1600 pixels, and JPEG quality 88. Model inference
uses bfloat16 on supported CUDA devices. The Exact MaxSim index stores normalized vectors in float16;
the SQLite embedding cache stores compressed float32 matrices in `.runtime/multi-vector/colsmol-500m-pages.sqlite3`.
Set `MIR_DEVICE` to select a device and `MIR_COLSMOL_500M_BATCH_SIZE` to override the default batch size of 4.

See the [visual Exact MaxSim run instructions](../pdf-solutions.md#visual-exact-maxsim) for the isolated smoke
test and full benchmark commands. GPU memory grows with the total page-token count, in addition to model
weights and inference activations.
