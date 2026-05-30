"""The grounding prompt is the contract that keeps the LLM honest; pin it."""

from ray_rag.serve.generate import build_messages, format_passages

_PASSAGES = [
    {"chunk_id": "ray_serve.md#0", "text": "Ray Serve composes models."},
    {"chunk_id": "rag_overview.md#1", "text": "Reranking sharpens the top results."},
]


def test_passages_are_labelled_by_citable_id():
    formatted = format_passages(_PASSAGES)
    assert "[ray_serve.md#0]" in formatted
    assert "[rag_overview.md#1]" in formatted


def test_system_instruction_is_cache_eligible_and_grounding():
    system, messages = build_messages("How does Ray Serve work?", _PASSAGES)
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "USING ONLY" in system[0]["text"]  # grounding constraint present
    # The variable passages + question travel in the user turn, not the cached block.
    assert "How does Ray Serve work?" in messages[0]["content"]
    assert "ray_serve.md#0" in messages[0]["content"]
