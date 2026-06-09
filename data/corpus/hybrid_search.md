# Hybrid Search

Dense retrieval matches on meaning: it embeds the query and the chunks into
vectors and compares them, so it can find a relevant passage that shares no
words with the question. Its weakness is the exact term — a specific identifier,
error code, or rare proper noun — which a semantic embedding can blur together
with similar-looking tokens.

Lexical search, such as BM25, has the opposite profile. It scores documents by
the query terms they actually contain, so it nails exact keywords but misses a
relevant passage phrased in different words. Hybrid search runs both and combines
their scores, aiming to keep the recall of dense matching and the precision of
lexical matching on exact terms.

The combination is typically done by normalising and summing the two scores, or
by fusing the two ranked lists. The result is a candidate set that is stronger
than either method alone, which a downstream reranker can then order precisely.
Hybrid search is about *which* candidates are retrieved, a separate concern from
how they are finally ranked.
