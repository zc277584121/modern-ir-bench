# BGE-M3 dense + learned sparse fusion

BGE-M3 exposes dense and learned-sparse representations from the same multilingual model. This Solution retrieves the shared 768-token chunks independently with both representations, collapses chunk hits to whole-document rankings, and fuses the two Top-50 lists.

The published Solution uses equal-weight reciprocal rank fusion with `k=60`. The persisted branch rankings are also available in Fusion Lab, where weights and fusion methods can be explored without rerunning BGE-M3.
