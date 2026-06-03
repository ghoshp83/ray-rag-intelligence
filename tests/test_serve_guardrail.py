"""The anti-fake-AI guarantee, pinned: an out-of-scope query is refused *before*
any retrieval or LLM call, and an in-scope query flows retrieve -> rerank ->
generate in that order. This is the thesis the README sells; without this test a
refactor that retrieved (or generated) before routing would silently break it.

The Ingress is a Ray Serve FastAPI deployment, so we test the underlying class
(`func_or_class`) with fake async deployment handles — no Ray cluster, no
ANTHROPIC_API_KEY, just the composition logic.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from ray_rag.serve.deployments import AskRequest, Ingress

_INGRESS_CLS = Ingress.func_or_class
_PASSAGES = [{"chunk_id": "a.md#0-ef0e58ff7481", "text": "t", "source": "a.md"}]


def _handles(intent: str):
    """Four fake deployment handles whose `.<method>.remote(...)` await to fixtures."""
    retriever = MagicMock()
    retriever.retrieve.remote = AsyncMock(return_value=_PASSAGES)
    reranker = MagicMock()
    reranker.rerank.remote = AsyncMock(return_value=_PASSAGES)
    router = MagicMock()
    router.route.remote = AsyncMock(return_value={"intent": intent, "confidence": 0.9})
    generator = MagicMock()
    generator.generate.remote = AsyncMock(
        return_value={
            "answer": "A",
            "sources": [{"chunk_id": "a.md#0-ef0e58ff7481", "source": "a.md"}],
        }
    )
    return retriever, reranker, router, generator


def test_out_of_scope_query_is_refused_before_retrieval_or_generation():
    retriever, reranker, router, generator = _handles("out_of_scope")
    ingress = _INGRESS_CLS(retriever, reranker, router, generator)

    result = asyncio.run(ingress.ask(AskRequest(query="something the corpus cannot ground")))

    assert result["intent"] == "out_of_scope"
    assert result["sources"] == []
    # The point of the guardrail: no retrieval, no rerank, no LLM call happened.
    assert retriever.retrieve.remote.called is False
    assert reranker.rerank.remote.called is False
    assert generator.generate.remote.called is False


def test_in_scope_query_flows_retrieve_rerank_generate_in_order():
    retriever, reranker, router, generator = _handles("factual")
    ingress = _INGRESS_CLS(retriever, reranker, router, generator)

    result = asyncio.run(ingress.ask(AskRequest(query="a real question")))

    assert result["intent"] == "factual"
    assert result["answer"] == "A"
    # Composition is wired correctly: rerank sees the retrieved candidates and
    # generation sees the reranked passages — not the raw query alone.
    assert reranker.rerank.remote.call_args.args[1] == _PASSAGES
    assert generator.generate.remote.call_args.args[1] == _PASSAGES


def test_health_endpoint_reports_ok():
    ingress = _INGRESS_CLS(*_handles("factual"))
    assert ingress.health() == {"status": "ok"}
