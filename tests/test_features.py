"""Reranker feature assembly: the trained XGBoost ranker scores on a fixed column
order (FEATURE_NAMES). If `features` emits columns in the wrong order — or the
dense cosine and cross-encoder score get swapped — the ranker silently scores on
the wrong signal and nDCG quietly degrades. These tests pin the column contract,
the lexical-overlap math, and the fail-loud on a missing key.

The cross-encoder is injected as a fake so the test needs no model download, no
GPU, and no network — it exercises only the assembly logic.
"""

import pytest

from ray_rag.models.features import FEATURE_NAMES, FeatureExtractor, lexical_overlap


def _extractor(ce_scores) -> FeatureExtractor:
    """A FeatureExtractor whose cross-encoder returns canned scores, no model load."""
    fx = object.__new__(FeatureExtractor)
    fake_ce = type("FakeCE", (), {"predict": lambda self, pairs: ce_scores})()
    fx._cross_encoder = fake_ce
    return fx


def test_feature_names_are_the_three_locked_signals():
    # The column contract the trained ranker depends on; reordering breaks it.
    assert FEATURE_NAMES == ["dense_cosine", "cross_encoder_score", "lexical_overlap"]


def test_lexical_overlap_is_fraction_of_query_tokens_present():
    # 2 of 3 query tokens ("ray", "serve") appear in the text -> 2/3.
    assert lexical_overlap("Ray Serve graph", "ray serve composes models") == pytest.approx(2 / 3)


def test_lexical_overlap_empty_query_is_zero_not_division_error():
    assert lexical_overlap("", "anything") == 0.0


def test_features_matrix_aligns_columns_to_feature_names():
    fx = _extractor(ce_scores=[0.7, 0.2])
    candidates = [
        {"text": "ray serve composes models", "score": 0.9},
        {"text": "unrelated text", "score": 0.1},
    ]
    matrix = fx.features("Ray Serve", candidates)
    assert matrix.shape == (2, 3)
    # Row 0: dense_cosine=score(0.9), cross_encoder_score=ce(0.7), lexical_overlap=2/2.
    assert matrix[0] == pytest.approx([0.9, 0.7, 1.0])
    # Row 1: dense cosine and CE must NOT be swapped — 0.1 then 0.2, overlap 0.
    assert matrix[1] == pytest.approx([0.1, 0.2, 0.0])


def test_features_empty_candidates_returns_empty_three_column_matrix():
    fx = _extractor(ce_scores=[])
    matrix = fx.features("anything", [])
    assert matrix.shape == (0, 3)


def test_features_missing_score_key_fails_loud():
    fx = _extractor(ce_scores=[0.5])
    with pytest.raises(KeyError):
        fx.features("q", [{"text": "no score key here"}])
