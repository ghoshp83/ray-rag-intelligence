"""Ray Serve deployment graph: retrieve -> rerank -> route -> generate.

Each stage is its own deployment so replicas and resources scale independently
(the embedder/index are CPU-light; the generator is the part that would move to
GPU under the documented vLLM scale-out). The ingress composes them and enforces
the routing guardrail: an `out_of_scope` query is refused before any retrieval
or LLM call, so the model never answers something the corpus cannot ground.
"""

from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve

from ray_rag.config import settings
from ray_rag.data.embed import Embedder
from ray_rag.data.index import VectorIndex
from ray_rag.models.intent import IntentClassifier
from ray_rag.models.reranker import Reranker
from ray_rag.observability import log_event
from ray_rag.serve.generate import generate_answer

api = FastAPI()


class AskRequest(BaseModel):
    query: str


@serve.deployment(ray_actor_options={"num_cpus": 1})
class Retriever:
    def __init__(self) -> None:
        self._embedder = Embedder(settings.embed_model)
        self._index = VectorIndex.load(settings.index_path)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        return self._index.search(self._embedder.encode([query]), top_k)[0]


@serve.deployment(ray_actor_options={"num_cpus": 1})
class RerankerDeployment:
    def __init__(self) -> None:
        self._reranker = Reranker.load(settings.reranker_path)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        return self._reranker.rerank(query, candidates, top_k)


@serve.deployment(ray_actor_options={"num_cpus": 1})
class Router:
    def __init__(self) -> None:
        self._clf = IntentClassifier.load(settings.intent_path, Embedder(settings.embed_model))

    def route(self, query: str) -> dict:
        intent, confidence = self._clf.predict(query)
        return {"intent": intent, "confidence": confidence}


# GPU scale-out: swap this for a vLLM-backed deployment with ray_actor_options
# {"num_gpus": 1} on Anyscale — see deploy/. The interface stays the same.
@serve.deployment(ray_actor_options={"num_cpus": 1})
class Generator:
    def __init__(self) -> None:
        from anthropic import Anthropic

        self._client = Anthropic()
        self._model = settings.llm_model

    def generate(self, query: str, passages: list[dict]) -> dict:
        answer = generate_answer(self._client, self._model, query, passages)
        sources = [{"chunk_id": p["chunk_id"], "source": p["source"]} for p in passages]
        return {"answer": answer, "sources": sources}


@serve.deployment
@serve.ingress(api)
class Ingress:
    def __init__(self, retriever, reranker, router, generator) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._router = router
        self._generator = generator

    @api.get("/health")
    def health(self) -> dict:
        return {"status": "ok"}

    @api.post("/ask")
    async def ask(self, body: AskRequest) -> dict:
        start = time.perf_counter()
        route = await self._router.route.remote(body.query)
        if route["intent"] == "out_of_scope":
            log_event(
                "serve",
                "ask",
                intent=route["intent"],
                confidence=route["confidence"],
                refused=True,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
            return {
                "intent": route["intent"],
                "answer": "This question is outside the scope of the indexed corpus.",
                "sources": [],
            }
        candidates = await self._retriever.retrieve.remote(body.query, settings.retrieve_top_k)
        reranked = await self._reranker.rerank.remote(body.query, candidates, settings.rerank_top_k)
        result = await self._generator.generate.remote(body.query, reranked)
        log_event(
            "serve",
            "ask",
            intent=route["intent"],
            confidence=route["confidence"],
            refused=False,
            n_sources=len(result["sources"]),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return {"intent": route["intent"], **result}
