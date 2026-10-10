# Voyage 4 Large

These Solutions call `voyage-4-large` with the correct `document` and `query`
input types, normalize candidate-pool semantics in the Task, and search exact
cosine similarity with a Milvus FLAT index.

One variant embeds full documents. The other embeds the benchmark-wide shared
chunks and collapses chunk hits to document rankings.

Voyage is a hosted service, so its model alias does not expose an immutable
weights revision. Each live result therefore records the API model ID, source
commit, and run time; this limitation is explicit rather than represented as a
locally pinned model revision.
