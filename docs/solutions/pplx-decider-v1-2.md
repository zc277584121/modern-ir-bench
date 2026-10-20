# Voyage 4 Large → PPLX Decider v1.2

Voyage 4 Large retrieves the top 50 candidate documents from shared 768-token chunks. PPLX Decider v1.2 then reads each complete document together with the query in an independent OpenRouter Decisions request and returns the probability that the document is relevant.

Documents are ordered by that probability. Equal probabilities retain the first-stage Voyage order. The published result records the exact dataset revision, source commit, model identifier, and aggregate API usage; request caches remain local and are not part of the release.

Model: `perplexity/pplx-decider-v1.2`  
Protocol: one query-document pair per request  
Candidate depth: 50  
Second-stage input: complete document text
