# Retrieval-Augmented Generation

Retrieval-augmented generation, or RAG, is an architecture that grounds a
language model's output in an external corpus of documents instead of relying
only on what the model memorised during training. At query time the system
retrieves relevant passages from the corpus and supplies them to the model as
context, so the generated answer can cite specific sources and stay current as
the corpus changes.

A RAG pipeline has several stages. First, documents are chunked into passages and
embedded into vectors that are stored in a vector index. When a query arrives, it
is embedded with the same model and the index returns the nearest passages by
similarity. Because dense retrieval favours recall over precision, a reranking
stage often follows: a cross-encoder model scores each candidate passage against
the query and reorders them, which sharply improves the quality of the top
results that the language model actually reads.

The final stage is generation. The language model receives the top reranked
passages and writes an answer constrained to that evidence, ideally citing the
passages it used. A trustworthy RAG system keeps the prediction work, namely
retrieval and reranking, in measurable models, and limits the language model to
the language task of turning ranked evidence into a grounded, cited answer. This
separation makes the system auditable: retrieval quality can be measured with
ranking metrics, and the answer's faithfulness to its cited sources can be scored
independently.
