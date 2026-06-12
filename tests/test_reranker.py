"""Reranker inference + the validation metric Ray Tune optimises.

`_mean_ndcg` is the *exact* number every Tune trial maximises to pick the
reranker's hyperparameters, so a sign error in its `argsort(-preds)` would
silently select a worse model with every other test still green. `rerank` is the
ordering the LLM actually reads: it must sort by learned score *descending* and
cap at top_k, or the grounded answer is built from the wrong evidence. Both are
exercised here with fakes (no XGBoost training, no model download).
"""

import math

import numpy as np

from ray_rag.models.reranker import (
    Reranker,
    _assemble,
    _mean_ndcg,
    build_ranking_examples,
)


def test_mean_ndcg_rewards_relevant_ranked_first():
    # One group of 3; preds rank the (only) relevant item top -> nDCG 1.0.
    preds = np.array([0.9, 0.1, 0.5])
    labels = np.array([1, 0, 0])
    assert _mean_ndcg(preds, labels, groups=[3], k=5) == 1.0


def test_mean_ndcg_penalises_relevant_ranked_last():
    # Same labels, but preds bury the relevant item -> discounted to rank 3.
    preds = np.array([0.1, 0.5, 0.9])
    labels = np.array([1, 0, 0])
    assert math.isclose(_mean_ndcg(preds, labels, groups=[3], k=5), 1.0 / math.log2(4))


def test_mean_ndcg_averages_across_groups():
    # Two queries concatenated: first perfect (1.0), second worst (1/log2 4).
    preds = np.array([0.9, 0.1, 0.1, 0.9])
    labels = np.array([1, 0, 1, 0])
    # Group 1 ranks the relevant item first (nDCG 1.0); group 2 buries it at
    # rank 2 of 2 (discounted by log2(3)). The metric is their mean.
    expected = (1.0 + 1.0 / math.log2(3)) / 2
    assert math.isclose(_mean_ndcg(preds, labels, groups=[2, 2], k=5), expected)


class _FakeExtractor:
    """Stands in for FeatureExtractor: shape-correct features, no cross-encoder."""

    def features(self, query, candidates):
        return np.zeros((len(candidates), 3), dtype=np.float32)


class _FakeBooster:
    """Returns scores by candidate position so we control the resulting order."""

    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=np.float32)

    def predict(self, dmatrix):
        return self._scores[: dmatrix.num_row()]


def _candidates(n):
    return [
        {"chunk_id": f"c{i}", "doc_id": f"d{i}", "text": f"t{i}", "score": 0.0} for i in range(n)
    ]


def test_rerank_orders_by_learned_score_descending():
    rr = Reranker(_FakeBooster([0.1, 0.9, 0.5]), _FakeExtractor())
    ranked = rr.rerank("q", _candidates(3), top_k=3)
    assert [c["chunk_id"] for c in ranked] == ["c1", "c2", "c0"]  # 0.9, 0.5, 0.1
    assert all(isinstance(c["rerank_score"], float) for c in ranked)
    assert math.isclose(ranked[0]["rerank_score"], 0.9, rel_tol=1e-6)  # float32 round-trip


def test_rerank_caps_at_top_k():
    rr = Reranker(_FakeBooster([0.1, 0.9, 0.5]), _FakeExtractor())
    ranked = rr.rerank("q", _candidates(3), top_k=2)
    assert [c["chunk_id"] for c in ranked] == ["c1", "c2"]  # top two by score


def test_rerank_empty_candidates_returns_empty():
    rr = Reranker(_FakeBooster([]), _FakeExtractor())
    assert rr.rerank("q", [], top_k=5) == []


# --- Training-data assembly: the rank:ndcg model is only as correct as its labels
# and per-query groups. A query whose retrieval is empty must be dropped (not
# emit a zero-length group XGBoost would choke on), labels must mark the relevant
# doc, and `groups` must equal each query's candidate count or the ranker learns
# across the wrong query boundaries. ------------------------------------------


class _FakeEmbedder:
    def encode(self, queries):
        return queries  # value unused; the fake index ignores it


class _ScriptedIndex:
    """Returns a preset candidate list per query, in call order (one search/query)."""

    def __init__(self, results):
        self._results = results
        self._i = 0

    def search(self, _vec, _k):
        out = self._results[self._i]
        self._i += 1
        return [out]


def _docs(*ids):
    return [{"chunk_id": f"{d}#0", "doc_id": d, "text": "t", "score": 0.0} for d in ids]


def test_build_ranking_examples_labels_by_relevant_doc_and_skips_empty_retrieval():
    labelled = [
        {"query": "q1", "relevant_docs": ["A"]},  # candidates A, B -> labels [1, 0]
        {"query": "q2", "relevant_docs": ["Z"]},  # no candidates -> skipped entirely
        {"query": "q3", "relevant_docs": ["C"]},  # candidate C -> labels [1]
    ]
    index = _ScriptedIndex([_docs("A", "B"), [], _docs("C")])
    examples = build_ranking_examples(index, _FakeEmbedder(), _FakeExtractor(), labelled, 30)

    assert len(examples) == 2  # q2 dropped, not emitted as an empty group
    assert [labels for _feats, labels in examples] == [[1, 0], [1]]


def test_assemble_groups_match_per_query_candidate_counts():
    # examples = [(feats(2,3), [1,0]), (feats(1,3), [1])] -> stacked X(3,3),
    # flat y of length 3, and groups [2, 1] (XGBoost's query boundaries).
    examples = [
        (np.zeros((2, 3), dtype=np.float32), [1, 0]),
        (np.zeros((1, 3), dtype=np.float32), [1]),
    ]
    X, y, groups = _assemble(examples)
    assert X.shape == (3, 3)
    assert list(y) == [1, 0, 1]
    assert groups == [2, 1]
