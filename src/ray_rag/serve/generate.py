"""Grounded generation: the LLM's only job. It must answer strictly from the
reranked passages and cite their ids — never from parametric knowledge.

`build_messages` is a pure function (no network) so the grounding contract — the
system instruction and how passages are presented — is unit-tested. The static
instruction is sent as a cache-eligible system block (prompt caching) since it
is identical across requests; the variable passages go in the user turn.
"""

from __future__ import annotations

INSTRUCTION = (
    "You are a careful document-grounded assistant. Answer the user's question "
    "USING ONLY the numbered sources provided. After each claim, cite the source "
    "id(s) it came from in square brackets, copied exactly as shown before each "
    "source, e.g. [ray_serve.md#0-ef0e58ff7481]. If the sources do not contain the "
    "answer, say you do not know — do not use outside knowledge and do not invent "
    "citations."
)


def format_passages(passages: list[dict]) -> str:
    return "\n\n".join(f"[{p['chunk_id']}] {p['text']}" for p in passages)


def build_messages(query: str, passages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (system_blocks, messages) for the Anthropic Messages API."""
    system = [{"type": "text", "text": INSTRUCTION, "cache_control": {"type": "ephemeral"}}]
    user = f"Sources:\n{format_passages(passages)}\n\nQuestion: {query}"
    messages = [{"role": "user", "content": user}]
    return system, messages


def generate_answer(client, model: str, query: str, passages: list[dict]) -> str:
    """Call Claude with the grounded prompt. Fails loud if the API errors."""
    system, messages = build_messages(query, passages)
    resp = client.messages.create(model=model, max_tokens=1024, system=system, messages=messages)
    return "".join(block.text for block in resp.content if block.type == "text")
