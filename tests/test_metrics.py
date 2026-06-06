"""Ranking metrics + lexical feature. These define the reranker's training
objective, so a wrong implementation would silently train the wrong model."""

import math

from ray_rag.eval.metrics import mrr, ndcg_at_k, recall_at_k
from ray_rag.models.features import lexical_overlap


def test_ndcg_perfect_ordering_is_one():
    assert ndcg_at_k([1, 1, 0, 0], k=4) == 1.0


def test_ndcg_penalises_relevant_item_ranked_lower():
    good = ndcg_at_k([1, 0, 0], k=3)  # relevant first
    bad = ndcg_at_k([0, 0, 1], k=3)  # relevant last
    assert good == 1.0
    assert bad < good
    assert math.isclose(bad, 1.0 / math.log2(4))  # gain discounted by rank-3 position


def test_ndcg_no_relevant_items_is_zero():
    assert ndcg_at_k([0, 0, 0], k=3) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert mrr([0, 0, 1, 1]) == 1.0 / 3
    assert mrr([0, 0, 0]) == 0.0


def test_recall_at_k_counts_distinct_relevant_docs_in_topk():
    # two relevant docs; only "a" is inside the top-2 window -> 1/2 recall.
    assert recall_at_k(["a", "x", "b"], relevant={"a", "b"}, k=2) == 0.5
    # both relevant docs present within k -> full recall.
    assert recall_at_k(["a", "b", "x"], relevant={"a", "b"}, k=3) == 1.0


def test_recall_at_k_dedupes_chunks_of_the_same_doc():
    # repeated chunks of "a" must not inflate recall past the single relevant doc.
    assert recall_at_k(["a", "a", "a"], relevant={"a", "b"}, k=3) == 0.5


def test_recall_at_k_no_relevant_docs_is_zero():
    assert recall_at_k(["a", "b"], relevant=set(), k=2) == 0.0


def test_lexical_overlap_is_fraction_of_query_tokens_present():
    assert lexical_overlap("ray serve graph", "ray serve deployment") == 2 / 3
    assert lexical_overlap("", "anything") == 0.0
