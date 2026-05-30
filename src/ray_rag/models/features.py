"""Reranker features: the interpretable signals the learned ranker combines.

Each (query, candidate) pair becomes a fixed vector. The cross-encoder is loaded
lazily and used *only* as a feature here (never as the final ranker), so the
ranking decision stays in a model we train. The lexical-overlap term is a pure
function so it is unit-tested without loading any model.
"""

from __future__ import annotations

import re

import numpy as np

FEATURE_NAMES = ["dense_cosine", "cross_encoder_score", "lexical_overlap"]
_TOKEN = re.compile(r"[a-z0-9]+")
_DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def lexical_overlap(query: str, text: str) -> float:
    """Fraction of query tokens present in the candidate (0..1)."""
    q = _tokens(query)
    if not q:
        return 0.0
    return len(q & _tokens(text)) / len(q)


class FeatureExtractor:
    def __init__(self, cross_encoder_model: str = _DEFAULT_CROSS_ENCODER):
        from sentence_transformers import CrossEncoder

        self._cross_encoder = CrossEncoder(cross_encoder_model)

    def features(self, query: str, candidates: list[dict]) -> np.ndarray:
        """(n_candidates, 3) matrix aligned to FEATURE_NAMES.

        Each candidate must carry `text` and `score` (the dense cosine from
        retrieval). A missing key is a wiring bug and should fail loud.
        """
        if not candidates:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self._cross_encoder.predict(pairs)
        rows = [
            (float(c["score"]), float(ce), lexical_overlap(query, c["text"]))
            for c, ce in zip(candidates, ce_scores, strict=True)
        ]
        return np.asarray(rows, dtype=np.float32)
