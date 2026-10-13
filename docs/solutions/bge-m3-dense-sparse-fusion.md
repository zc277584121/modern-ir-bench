# BGE-M3 dense + learned sparse fusion

BGE-M3 exposes dense and learned-sparse representations from the same multilingual model. These Solutions retrieve the shared 768-token chunks independently with both representations, collapse chunk hits to whole-document rankings, and fuse the two Top-50 lists.

The RRF variant uses `k=60`. The weighted variants apply per-query min-max normalization before assigning dense/sparse weights of 25/75, 50/50, or 75/25. All variants reuse the same persisted branch rankings, so changing a fusion weight does not rerun BGE-M3.
