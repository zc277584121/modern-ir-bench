# ColPali v1.3 · page exact MaxSim

This OCR-free Solution renders PDF pages as images, encodes them with `vidore/colpali-v1.3`, and exhaustively
scores every page with Exact MaxSim. Queries and pages contain multiple 128-dimensional vectors; each PDF
receives its best page score.

This Solution supports only English in this benchmark. Run it with `--language en`;
its scores are eligible for the English breakdown and excluded from bilingual overall and Chinese rankings.

The [official model configuration](https://huggingface.co/vidore/colpali-v1.3) is loaded through the shared
`SentenceTransformersMultiVectorEmbedding` component using Sentence Transformers `MultiVectorEncoder`.
The pinned revision is `b5c6dd62125326e6f0b540c1f7b36e901cdc0a11`. Its adapter loads the
`vidore/colpaligemma-3b-pt-448-base` backbone. The published configuration supplies the training-time
query prefix, projection, adapter weight mapping, and `token_type_ids` used for attention masking.

The shared PDF renderer uses scale 2, a maximum side of 1600 pixels, and JPEG quality 88. Model inference
uses bfloat16 on supported CUDA devices. The Exact MaxSim index stores normalized vectors in float16;
the SQLite embedding cache stores compressed float32 matrices in `.runtime/multi-vector/colpali-v1.3-pages.sqlite3`.
Set `MIR_DEVICE` to select a device and `MIR_COLPALI_V1_3_BATCH_SIZE` to override the default batch size of 1.

See the [visual Exact MaxSim run instructions](../pdf-solutions.md#visual-exact-maxsim) for the isolated smoke
test and full benchmark commands. GPU memory grows with the total page-token count, in addition to model
weights and inference activations.
