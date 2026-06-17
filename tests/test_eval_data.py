"""Data contracts for the held-out reranker relevance sets.

The reranker's headline nDCG/MRR is only honest if the test queries are unseen:
the train set tunes+fits the ranker, the test set scores it. These pin the
invariants that keep that true, so a future edit to the JSONL files fails a test
here rather than silently turning the held-out number back into a train-set one.
"""

from __future__ import annotations

from ray_rag.config import settings
from ray_rag.data.chunk import load_documents
from ray_rag.models.train import load_jsonl


def test_train_and_test_queries_are_disjoint():
    # The 2026-06-05 split exists precisely to kill train-set leakage (the prior
    # 0.98 nDCG was a train-set number). If a query landed in both files, the
    # ranker would be scored on a query it trained on and the held-out figure
    # would quietly inflate while every metric test stayed green. Pin disjointness.
    train_q = {ex["query"] for ex in load_jsonl(settings.eval_train_path)}
    test_q = {ex["query"] for ex in load_jsonl(settings.eval_path)}
    assert train_q.isdisjoint(test_q), sorted(train_q & test_q)


def test_relevant_docs_reference_real_corpus_docs():
    # `relevant_docs` are matched against retrieved candidates' doc_ids, which are
    # corpus file paths relative to the corpus root (data/chunk.load_documents). A
    # typo'd or renamed label can never match a retrieved doc, so it silently
    # depresses recall/nDCG with no error — the relevant doc looks un-retrievable.
    # Pin every label to an actual corpus doc so a drift fails loud here.
    corpus_docs = {doc_id for doc_id, _src, _text in load_documents(settings.corpus_path)}
    labelled = load_jsonl(settings.eval_train_path) + load_jsonl(settings.eval_path)
    referenced = {doc for ex in labelled for doc in ex["relevant_docs"]}
    assert referenced <= corpus_docs, sorted(referenced - corpus_docs)
