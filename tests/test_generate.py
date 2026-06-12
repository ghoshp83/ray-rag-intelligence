"""The grounding prompt is the contract that keeps the LLM honest; pin it."""

from ray_rag.serve.generate import build_messages, format_passages, generate_answer

# Real `#idx-hash` chunk ids — the label the model must copy verbatim.
_PASSAGES = [
    {"chunk_id": "ray_serve.md#0-ef0e58ff7481", "text": "Ray Serve composes models."},
    {"chunk_id": "rag_overview.md#1-8ed05e34c9a0", "text": "Reranking sharpens the top results."},
]


def test_passages_are_labelled_by_citable_id():
    formatted = format_passages(_PASSAGES)
    assert "[ray_serve.md#0-ef0e58ff7481]" in formatted
    assert "[rag_overview.md#1-8ed05e34c9a0]" in formatted


def test_system_instruction_is_cache_eligible_and_grounding():
    system, messages = build_messages("How does Ray Serve work?", _PASSAGES)
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "USING ONLY" in system[0]["text"]  # grounding constraint present
    # The variable passages + question travel in the user turn, not the cached block.
    assert "How does Ray Serve work?" in messages[0]["content"]
    assert "ray_serve.md#0-ef0e58ff7481" in messages[0]["content"]


class _Block:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        self.text = text


class _Messages:
    def __init__(self, blocks: list) -> None:
        self._blocks = blocks
        self.seen: dict | None = None

    def create(self, **kwargs):
        self.seen = kwargs
        return type("Resp", (), {"content": self._blocks})()


class _FakeClient:
    def __init__(self, blocks: list) -> None:
        self.messages = _Messages(blocks)


def test_generate_answer_joins_only_text_blocks_in_order():
    # A Claude response can interleave non-text blocks (tool_use / thinking); the
    # answer must concatenate the text blocks in order and skip the rest, never
    # reach for `.text` on a block that has none.
    client = _FakeClient(
        [_Block("text", "Ray Serve "), _Block("tool_use"), _Block("text", "composes models.")]
    )
    answer = generate_answer(client, "claude-opus-4-8", "How?", _PASSAGES)
    assert answer == "Ray Serve composes models."
    assert client.messages.seen["model"] == "claude-opus-4-8"  # model id passed through
