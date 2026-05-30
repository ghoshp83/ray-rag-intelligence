"""Grounding score must flag citations the model was never given (hallucinated provenance)."""

from ray_rag.eval.grounding import extract_citations, grounding_score

_PROVIDED = ["ray_serve.md#0", "rag_overview.md#1"]


def test_extract_citations_finds_doc_index_ids():
    assert extract_citations("Serve composes models [ray_serve.md#0].") == ["ray_serve.md#0"]


def test_all_citations_valid_when_grounded():
    s = grounding_score("Answer [ray_serve.md#0] and [rag_overview.md#1].", _PROVIDED)
    assert s["has_citation"] is True
    assert s["valid_fraction"] == 1.0


def test_invalid_citation_lowers_fraction():
    # one real id, one fabricated id not among provided passages
    s = grounding_score("Real [ray_serve.md#0] and fake [made_up.md#9].", _PROVIDED)
    assert s["n_citations"] == 2
    assert s["valid_fraction"] == 0.5


def test_uncited_answer_is_flagged():
    s = grounding_score("An answer with no citations at all.", _PROVIDED)
    assert s["has_citation"] is False
    assert s["valid_fraction"] == 1.0  # nothing cited -> nothing invalid
