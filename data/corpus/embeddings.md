# Text Embeddings

An embedding model maps a piece of text to a fixed-length vector of numbers so
that texts with similar meaning land close together in vector space. Modern
sentence-embedding models such as the BGE and sentence-transformers families are
trained on large collections of paired sentences so that semantic similarity, not
surface word overlap, drives the distance between vectors. This is what lets a
retrieval system match a question to a passage that answers it even when they
share few words.

For retrieval the query and the documents must be embedded by the identical
model, because two different models produce vectors that are not comparable. The
vectors are typically normalised to unit length so that the inner product between
two of them equals their cosine similarity, a value between minus one and one
where higher means more similar.

Embedding is an inference workload that batches well and runs acceptably on CPU
for small corpora, which makes it a natural fit for distributed batch processing:
the model is loaded once per worker and applied to many chunks in parallel. The
cost of embedding the corpus is paid once at indexing time, while queries are
embedded one at a time at serving time.
