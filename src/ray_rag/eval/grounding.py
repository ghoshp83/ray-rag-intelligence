"""Grounding / citation faithfulness — pure scoring, unit-testable without an LLM.

A grounded answer cites only chunk ids that were actually supplied to the model.
We measure two things: whether the answer cited anything at all, and what
fraction of its citations are valid (present in the provided passages). An
answer that cites ids it was never given is hallucinating provenance — the
exact failure this project exists to catch.
"""

from __future__ import annotations

import re

# Chunk ids are `doc#idx-hash` (see data/chunk.py), e.g. `[ray_serve.md#0-96161c49338c]`.
# A model often groups several in one bracket (`[a#0-h, b#1-h]`) as well as using
# separate brackets, so pull every bracketed run, then split it on commas /
# semicolons / whitespace — otherwise a grouped, well-grounded citation is silently
# dropped and the grounding metric undercounts. Membership in the provided ids then
# judges validity: a malformed near-citation (e.g. the hash dropped) is still
# extracted but counts invalid, while a bare `#heading` or a link label is no id and
# is ignored here.
_BRACKET = re.compile(r"\[([^\[\]]+)\]")


def _looks_like_id(token: str) -> bool:
    # A `doc#idx-hash` id has non-empty text on both sides of a `#`; this excludes
    # a bare `#heading` (starts with `#`) and bracketed prose with no `#`.
    return "#" in token and not token.startswith("#") and not token.endswith("#")


def extract_citations(answer: str) -> list[str]:
    """Pull `[doc#idx-hash]`-style citation ids from an answer, in order.

    Ids grouped in one bracket (comma / semicolon / space separated) are each
    returned; bracketed text that is not an id (a `#heading`, a link label) is
    ignored.
    """
    return [
        token
        for body in _BRACKET.findall(answer)
        for token in re.split(r"[\s,;]+", body)
        if _looks_like_id(token)
    ]


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
