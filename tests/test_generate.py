"""The grounding prompt is the contract that keeps the LLM honest; pin it."""

import re

from ray_rag.data.chunk import _chunk_id
from ray_rag.eval.grounding import extract_citations
from ray_rag.serve.generate import (
    INSTRUCTION,
    build_messages,
    format_passages,
    generate_answer,
)

# The id shown verbatim in INSTRUCTION as the format the model must copy.
_EXAMPLE_ID = "ray_serve.md#0-ef0e58ff7481"
# A real chunk id is `doc#idx-hash` with a 12-hex sha1 prefix (data/chunk.py).
_CHUNK_ID_SHAPE = re.compile(r"^[^#]+#\d+-[0-9a-f]{12}$")

# Real `#idx-hash` chunk ids — the label the model must copy verbatim.
_PASSAGES = [
    {"chunk_id": "ray_serve.md#0-ef0e58ff7481", "text": "Ray Serve composes models."},
    {"chunk_id": "rag_overview.md#1-8ed05e34c9a0", "text": "Reranking sharpens the top results."},
]


def test_instruction_example_round_trips_through_the_citation_extractor():
    # INSTRUCTION teaches the model to copy ids "exactly as shown" via this one
    # example, and the grounding eval scores answers by extracting bracketed ids.
    # The two must agree: if the example drifts to a shape extract_citations
    # won't recover (the historic bug dropped the `-hash` suffix), the model
    # copies an un-scorable id and grounding silently measures nothing while
    # every other test stays green. Pin the prompt <-> extractor contract.
    assert _EXAMPLE_ID in INSTRUCTION
    assert extract_citations(INSTRUCTION) == [_EXAMPLE_ID]


def test_instruction_example_matches_the_real_chunk_id_shape():
    # The taught example must look like an id data/chunk.py actually mints, not a
    # simplified placeholder, so what the model copies matches the ids it is given
    # by format_passages and the ids the index stores.
    assert _CHUNK_ID_SHAPE.match(_EXAMPLE_ID)
    assert _CHUNK_ID_SHAPE.match(_chunk_id("ray_serve.md", 0, "some chunk text"))
    rendered = format_passages([{"chunk_id": _EXAMPLE_ID, "text": "body"}])
    assert extract_citations(rendered) == [_EXAMPLE_ID]


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
