"""The grounding eval's orchestration, pinned without an LLM call.

`evaluate_grounding` retrieves, reranks, generates, then scores the answer's
citations against the *reranked* passage ids — not the raw retrieved candidates.
A fake client lets us pin that wiring and the aggregation key-free; the real
metric still needs ANTHROPIC_API_KEY (harness `main()`), and that skip stays loud.
"""

from __future__ import annotations

from ray_rag.eval.harness import evaluate_grounding


class _FakeEmbedder:
    def encode(self, queries):  # noqa: ANN001 - duck-typed, value unused by fakes
        return queries


class _FakeIndex:
    """Retrieves two candidate docs, a and b."""

    def search(self, _vec, _k):  # noqa: ANN001
        return [
            [
                {"chunk_id": "a.md#0-aaaa", "doc_id": "a.md", "source": "a.md", "text": "t"},
                {"chunk_id": "b.md#0-bbbb", "doc_id": "b.md", "source": "b.md", "text": "t"},
            ]
        ]


class _TopOneReranker:
    """Keeps only the top candidate — so a citation to the dropped doc is invalid."""

    def rerank(self, _query, candidates, _k):  # noqa: ANN001
        return candidates[:1]


class _Messages:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **kwargs):
        block = type("Block", (), {"type": "text", "text": self._text})()
        return type("Resp", (), {"content": [block]})()


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = _Messages(text)


def test_grounding_scored_against_reranked_ids_not_retrieved_candidates():
    # The answer cites both retrieved docs, but the reranker dropped b — so b's
    # citation must count invalid (1 of 2 valid). Had grounding been scored against
    # the *retrieved* candidates (a, b), both would count and the drop would hide.
    labelled = [{"query": "q", "relevant_docs": ["a.md"]}]
    answer = "A claim [a.md#0-aaaa] and another [b.md#0-bbbb]."
    out = evaluate_grounding(
        _FakeIndex(), _FakeEmbedder(), _TopOneReranker(), labelled, client=_FakeClient(answer)
    )
    assert out["mean_valid_citation_fraction"] == 0.5
    assert out["answers_with_citation"] == 1.0
    assert out["n_queries"] == 1
