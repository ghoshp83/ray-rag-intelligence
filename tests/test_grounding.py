"""Grounding score must flag citations the model was never given (hallucinated provenance)."""

from ray_rag.eval.grounding import extract_citations, grounding_score

# Real chunk ids carry the `#idx-hash` suffix (data/chunk.py); the scorer must
# work on that exact format, not a simplified `doc#idx`.
_PROVIDED = ["ray_serve.md#0-ef0e58ff7481", "rag_overview.md#1-8ed05e34c9a0"]


def test_extract_citations_finds_real_hashed_ids():
    answer = "Serve composes models [ray_serve.md#0-ef0e58ff7481]."
    assert extract_citations(answer) == ["ray_serve.md#0-ef0e58ff7481"]


def test_all_citations_valid_when_grounded():
    s = grounding_score(
        "Answer [ray_serve.md#0-ef0e58ff7481] and [rag_overview.md#1-8ed05e34c9a0].",
        _PROVIDED,
    )
    assert s["has_citation"] is True
    assert s["valid_fraction"] == 1.0


def test_invalid_citation_lowers_fraction():
    # one real id, one fabricated id not among provided passages
    s = grounding_score(
        "Real [ray_serve.md#0-ef0e58ff7481] and fake [made_up.md#9-000000000000].",
        _PROVIDED,
    )
    assert s["n_citations"] == 2
    assert s["valid_fraction"] == 0.5


def test_hash_dropped_citation_counts_invalid():
    # The model citing `doc#idx` without the real hash is not a provided id;
    # it must be extracted (so it counts) yet judged invalid (it lowers the score).
    s = grounding_score("Looks right but [ray_serve.md#0] omits the hash.", _PROVIDED)
    assert s["n_citations"] == 1
    assert s["valid_fraction"] == 0.0


def test_uncited_answer_is_flagged():
    s = grounding_score("An answer with no citations at all.", _PROVIDED)
    assert s["has_citation"] is False
    assert s["valid_fraction"] == 1.0  # nothing cited -> nothing invalid
