"""Ranking metrics — pure functions, the single definition used by both the
reranker's training objective and the eval harness.

They take a list of relevance labels *in predicted rank order* (index 0 = the
item the model ranked first). Keeping one implementation means the number the
trainer optimises and the number the eval reports cannot silently diverge.
"""

from __future__ import annotations

import math


def dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_relevances: list[float], k: int) -> float:
    """Normalised DCG@k. Returns 0.0 when no relevant item exists (ideal DCG=0)."""
    ideal = dcg(sorted(ranked_relevances, reverse=True)[:k])
    if ideal == 0.0:
        return 0.0
    return dcg(ranked_relevances[:k]) / ideal


def mrr(ranked_relevances: list[float]) -> float:
    """Reciprocal rank of the first relevant (>0) item; 0.0 if none."""
    for i, rel in enumerate(ranked_relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0
