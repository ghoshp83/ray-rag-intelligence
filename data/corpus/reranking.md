# Reranking Retrieved Results

Dense retrieval is tuned for recall: it casts a wide net and returns many
candidate chunks, some only loosely relevant. A reranking stage reorders those
candidates so the most relevant ones rise to the top, because the generation step
reads only the first few and their ordering decides what evidence the answer is
built on. Improving the order at the top is what a reranker is for.

A learned-to-rank reranker is a model trained on labelled relevance to predict a
ranking score from interpretable features: the dense similarity between query and
chunk, a cross-encoder relevance score, and lexical overlap. The cross-encoder
runs the query and chunk through a transformer together, which is more accurate
than comparing independent embeddings but too slow to apply to a whole corpus, so
it is used only as a feature over the shortlist. Training the ranker ourselves
means its quality is a measured number we can move, optimised to maximise ranking
metrics such as nDCG and MRR rather than a fixed black box.

Crucially, ranking is a prediction task owned by a trained model, not by a large
language model. Using an LLM to order results would be slow, costly, and
unmeasurable; the trained ranker is faster, cheaper, and evaluated on a held-out
relevance set. The language model is reserved for what it is genuinely best at:
turning the top-ranked evidence into a grounded natural-language answer.
