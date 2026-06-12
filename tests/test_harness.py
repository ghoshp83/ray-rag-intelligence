"""Tests for the eval harness's per-query uplift diagnostic.

`uplift_summary` exists so a flat dense->reranked average cannot hide a reranker
that helps and hurts queries in equal measure. These pin that the counts reflect
the sign of each per-query delta and that float noise reads as "tied", not as a
spurious win or loss — otherwise the diagnostic would mislabel a parity result.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ray_rag.eval.harness import evaluate_intent, evaluate_reranker, uplift_summary


def _rows(deltas: list[float]) -> list[dict]:
    return [{"delta": d} for d in deltas]


def test_counts_match_delta_signs():
    s = uplift_summary(_rows([0.2, -0.1, 0.0, 0.3, -0.4]))
    assert (s["n_improved"], s["n_regressed"], s["n_tied"]) == (2, 2, 1)


def test_float_noise_counts_as_tied_not_a_win_or_loss():
    # Deltas within tolerance are rounding noise, not real movement.
    s = uplift_summary(_rows([1e-12, -1e-12]))
    assert (s["n_improved"], s["n_regressed"], s["n_tied"]) == (0, 0, 2)


def test_empty_is_all_zero():
    assert uplift_summary([]) == {"n_improved": 0, "n_regressed": 0, "n_tied": 0}


def _docs(*ids: str) -> list[dict]:
    return [{"doc_id": i} for i in ids]


class _ScriptedStage:
    """Returns preset result lists, one per call, in order — fakes a stage that
    is queried once per labelled example so a test can fix dense and reranked
    orderings independently of any model."""

    def __init__(self, results: list[list[dict]]) -> None:
        self._results = results
        self._i = 0

    def _next(self) -> list[dict]:
        out = self._results[self._i]
        self._i += 1
        return out


class _FakeEmbedder:
    def encode(self, queries):  # noqa: ANN001 - duck-typed, value unused by fakes
        return queries


class _FakeIndex(_ScriptedStage):
    def search(self, _vec, _k):  # noqa: ANN001
        return [self._next()]


class _FakeReranker(_ScriptedStage):
    def rerank(self, _query, _candidates, _n):  # noqa: ANN001
        return self._next()


def test_diagnostic_credits_reranked_order_not_dense_order():
    # q1: dense ranks the relevant doc A second; the reranker lifts it to first
    # -> reranked nDCG must beat dense nDCG (improved). q2: dense already optimal
    # and the reranker leaves it -> tied. The counts must reflect the *reranked*
    # ordering, which is the claim the README's uplift breakdown makes.
    labelled = [
        {"query": "q1", "relevant_docs": ["A"]},
        {"query": "q2", "relevant_docs": ["X"]},
    ]
    index = _FakeIndex([_docs("B", "A", "C"), _docs("X", "Y", "Z")])
    reranker = _FakeReranker([_docs("A", "B", "C"), _docs("X", "Y", "Z")])

    out = evaluate_reranker(index, _FakeEmbedder(), reranker, labelled, k=5)

    assert (out["n_improved"], out["n_regressed"], out["n_tied"]) == (1, 0, 1)
    assert out["reranked_ndcg"] > out["dense_ndcg"]
    assert out["reranked_ndcg"] == 1.0  # both queries perfectly ordered after rerank
    assert out["per_query"][0]["delta"] > 0  # q1 lifted
    assert abs(out["per_query"][1]["delta"]) < 1e-9  # q2 unchanged


class _OneHotEmbedder:
    """Maps each query to a one-hot vector keyed by its leading intent token, so
    the three classes are linearly separable and the held-out F1 is deterministic
    — these tests pin the harness's wiring, not sklearn's accuracy."""

    _COLS = {"factual": 0, "summarize": 1, "out_of_scope": 2}

    def encode(self, queries):  # noqa: ANN001 - duck-typed
        return np.array([np.eye(3)[self._COLS[q.split()[0]]] for q in queries], dtype=np.float32)


def _intent_labelled() -> list[dict]:
    intents = ("factual", "summarize", "out_of_scope")
    return [{"query": f"{i} q{n}", "intent": i} for i in intents for n in range(6)]


def test_evaluate_intent_does_not_fit_or_mutate_the_shipped_classifier():
    # The harness must score the *deployed* model's tuned params via clone+refit on
    # a train split — never fit the passed-in classifier in place, which would both
    # leak the held-out rows into the fit and mutate the artifact we ship.
    clf = LogisticRegression(C=0.123, max_iter=1000)
    evaluate_intent(_intent_labelled(), _OneHotEmbedder(), clf)
    assert not hasattr(clf, "coef_")  # untouched: clone() isolated the passed clf


def test_evaluate_intent_reports_per_class_f1_keyed_by_sorted_labels():
    out = evaluate_intent(_intent_labelled(), _OneHotEmbedder(), LogisticRegression(max_iter=1000))
    # sklearn sorts labels alphabetically — NOT the INTENTS declaration order.
    assert list(out["per_class_f1"]) == ["factual", "out_of_scope", "summarize"]
    assert out["n_test"] == 6  # 18 samples, test_size=0.3 -> ceil(5.4) = 6
    assert out["holdout_macro_f1"] == 1.0  # one-hot features are perfectly separable
