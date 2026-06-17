"""Data contracts for the held-out reranker relevance sets.

The reranker's headline nDCG/MRR is only honest if the test queries are unseen:
the train set tunes+fits the ranker, the test set scores it. These pin the
invariants that keep that true, so a future edit to the JSONL files fails a test
here rather than silently turning the held-out number back into a train-set one.
"""

from __future__ import annotations

from ray_rag.config import settings
from ray_rag.models.train import load_jsonl


def test_train_and_test_queries_are_disjoint():
    # The 2026-06-05 split exists precisely to kill train-set leakage (the prior
    # 0.98 nDCG was a train-set number). If a query landed in both files, the
    # ranker would be scored on a query it trained on and the held-out figure
    # would quietly inflate while every metric test stayed green. Pin disjointness.
    train_q = {ex["query"] for ex in load_jsonl(settings.eval_train_path)}
    test_q = {ex["query"] for ex in load_jsonl(settings.eval_path)}
    assert train_q.isdisjoint(test_q), sorted(train_q & test_q)
