# Vector Search and FAISS

A vector index stores the embedding of every document chunk and answers a
nearest-neighbour query: given a query vector, return the stored vectors closest
to it. FAISS is a widely used library for this. Its simplest index, a flat index,
holds every vector and scores the query against all of them, which is exact but
costs time linear in the corpus size. For the small corpora a flat index is the
right choice because it returns the true top results with no approximation error.

When vectors are normalised to unit length, an inner-product flat index ranks by
cosine similarity, so the highest-scoring entries are the most semantically
similar chunks. The index returns the positions of the matches together with
their scores; those positions must be kept aligned with a parallel list of chunk
metadata, or the system will return text that does not correspond to the vector
it actually matched.

Larger corpora trade exactness for speed using approximate nearest-neighbour
indexes that cluster or quantise the vectors, searching only the most promising
regions. The metric that matters there is recall at k: the fraction of the true
nearest neighbours the approximate search still returns. Retrieval deliberately
favours recall so that a later reranking stage can refine precision at the top.
