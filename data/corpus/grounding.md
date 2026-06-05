# Grounded Generation and Citations

A trustworthy retrieval-augmented system never lets the language model answer from
its own memory. Instead it grounds every answer in the chunks retrieved and
reranked from the corpus, and it asks the model to cite the specific chunk
identifiers it used. Grounding is the defence against hallucination: if the
evidence does not support a claim, the claim should not appear in the answer.

Citation faithfulness can be measured rather than trusted. After generation, the
system extracts the chunk identifiers the model cited and checks each against the
set of chunks that were actually supplied. A citation that names a chunk never
provided is invalid and counts against the score; an answer with no citation at
all is a separate failure to catch. Reporting the fraction of valid citations and
the fraction of answers that cite anything turns grounding from a claim into a
number that can regress in tests.

When a query falls outside what the corpus covers, the honest response is to
refuse rather than to generate an unsupported answer. Routing the query first and
declining out-of-scope questions before any retrieval or generation keeps the
system from fabricating evidence it does not have, which is the same principle as
grounding applied one step earlier in the pipeline.
