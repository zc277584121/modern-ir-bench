# Voyage 4 Large → GPT-6 Luna Decisions

Voyage 4 Large retrieves the top 50 candidate documents from shared 768-token chunks. GPT-6 Luna Decisions evaluates them in five requests of ten isolated relevance questions. Each question contains one complete candidate document, while the query is shared state.

Documents are ordered by their individual relevance probabilities. Equal probabilities retain the first-stage Voyage order. The published result records the exact dataset revision, source commit, model identifier, and aggregate API usage; request caches remain local and are not part of the release.

Model: `openai/gpt-6-luna-decisions`  
Protocol: 10 isolated document questions per request  
Candidate depth: 50  
Second-stage input: complete document text
