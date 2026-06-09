# Fine-Tuning Versus Retrieval

Fine-tuning adapts a language model by continuing to train its weights on
domain examples, so the new knowledge or style is baked into the model itself.
It is the right tool when you need to change *how* a model behaves — its tone,
format, or a narrow skill — and you have enough labelled examples to train on.

Retrieval-augmented generation takes the opposite approach: the model weights
stay fixed and the relevant facts are fetched from a corpus at question time and
placed in the prompt. This is the better fit when the knowledge changes often or
must be traceable to a source, because updating the corpus is cheaper than
retraining and every answer can cite where its evidence came from.

The two are not exclusive. A system can fine-tune a model for a domain's style
and still retrieve current facts for each query. But for keeping answers
accurate and grounded as information changes, retrieval is usually the first
lever to reach for, and training the model's weights is reserved for behaviour
that retrieval cannot supply.
