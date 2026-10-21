# ColSmol 256M · page exact MaxSim

This OCR-free Solution renders each PDF page as an image and encodes it with the pinned `vidore/colSmol-256M`
checkpoint. Queries and pages are represented by multiple 128-dimensional vectors. Retrieval exhaustively scores every
page with MaxSim and collapses page hits to PDF document identifiers.

The exhaustive index stores vectors in float16 to fit the complete page corpus on a 12 GB GPU. Candidate generation
and approximate nearest-neighbor search are not used.
