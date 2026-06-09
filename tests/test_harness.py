"""Tests for the eval harness's per-query uplift diagnostic.

`uplift_summary` exists so a flat dense->reranked average cannot hide a reranker
that helps and hurts queries in equal measure. These pin that the counts reflect
the sign of each per-query delta and that float noise reads as "tied", not as a
spurious win or loss — otherwise the diagnostic would mislabel a parity result.
"""

from __future__ import annotations

from ray_rag.eval.harness import uplift_summary


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
