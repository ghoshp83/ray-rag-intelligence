"""Grounding / citation faithfulness — pure scoring, unit-testable without an LLM.

A grounded answer cites only chunk ids that were actually supplied to the model.
We measure two things: whether the answer cited anything at all, and what
fraction of its citations are valid (present in the provided passages). An
answer that cites ids it was never given is hallucinating provenance — the
exact failure this project exists to catch.
"""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[([^\]\s]+#\d+)\]")


def extract_citations(answer: str) -> list[str]:
    """Pull `[doc#idx]`-style citation ids from an answer, in order."""
    return _CITATION.findall(answer)


def grounding_score(answer: str, provided_ids: list[str]) -> dict:
    """has_citation: cited >=1 source; valid_fraction: share of citations that
    were actually provided (1.0 when there are no citations to invalidate)."""
    cited = extract_citations(answer)
    allowed = set(provided_ids)
    valid = [c for c in cited if c in allowed]
    return {
        "has_citation": len(cited) > 0,
        "valid_fraction": (len(valid) / len(cited)) if cited else 1.0,
        "n_citations": len(cited),
    }
