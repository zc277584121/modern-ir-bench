# BGE-M3

The dense and learned-sparse Solutions share one pinned BGE-M3 encoder at
revision `5617a9f61b028005a4858fdac845db406aefb181`. They use the model's
dedicated corpus and query encoding paths and reproduce both vector families
from the same inference pass.

Both variants use the benchmark-wide shared chunks. Dense vectors use exact
Milvus cosine search; learned-sparse vectors use a Milvus sparse inverted index
with inner-product scoring. Chunk hits are collapsed to document rankings.
