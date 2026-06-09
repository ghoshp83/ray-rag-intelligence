# Chunking Documents for Retrieval

Before a corpus can be embedded it must be split into chunks, because a whole
document is usually too long to embed as one vector and too coarse to retrieve
precisely. Chunking decides the unit of retrieval: each chunk becomes one
embedding and one candidate that retrieval can return, so the chunk boundary
shapes what evidence an answer can be built from.

Chunk size is a trade-off. Chunks that are too large dilute the embedding with
several topics and return irrelevant text alongside the relevant sentence;
chunks that are too small lose the surrounding context needed to make sense of a
passage. A common compromise is a few hundred tokens with a small overlap
between neighbours, so a fact that straddles a boundary still appears whole in at
least one chunk.

Splitting on natural structure — paragraphs or headings — tends to beat cutting
at a fixed character count, because it keeps coherent ideas together. Chunking is
a preprocessing choice that happens before any vector is computed.
