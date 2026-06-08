"""Measure Serve request-path latency — the trained-ML stages of the graph.

Times the per-query `route -> retrieve -> rerank` path (the order the Serve
ingress runs) in-process and reports p50 / p95 / mean per stage and end-to-end,
so latency is read per component, not as one opaque number. This is the part the
trained models own; it is CPU-only and needs no API key.

The `generate` stage is **excluded on purpose**: it is a network call to the
external LLM API, so its latency is dominated by the provider and the prompt
size, not by this code — timing it here would measure the API, not the graph.
The README states this exclusion.

The bundled eval queries are few, so one warm-up pass (discarded — it pays the
model-load cost) runs before the timed passes; we repeat the query set so the
percentiles have enough samples to be meaningful. CPU-only, no GPU, no API key.

Run: `python scripts/measure_latency.py` (or `make latency`).
"""

from __future__ import annotations

import time

import numpy as np

from ray_rag.config import settings
from ray_rag.data.embed import Embedder
from ray_rag.data.index import VectorIndex
from ray_rag.models.intent import IntentClassifier
from ray_rag.models.reranker import Reranker
from ray_rag.models.train import load_jsonl

REPEATS = 20  # repeat the query set so p50/p95 have enough samples


def _percentiles(samples_ms: list[float]) -> tuple[float, float, float]:
    arr = np.array(samples_ms)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), float(arr.mean())


def main() -> None:
    embedder = Embedder(settings.embed_model)
    index = VectorIndex.load(settings.index_path)
    reranker = Reranker.load(settings.reranker_path)
    classifier = IntentClassifier.load(settings.intent_path, embedder)
    queries = [ex["query"] for ex in load_jsonl(settings.eval_path)]
    if not queries:
        raise ValueError("no eval queries — is the held-out set empty?")

    stages = {"route": [], "retrieve": [], "rerank": [], "total": []}  # type: dict[str, list[float]]

    def one(query: str, record: bool) -> None:
        t0 = time.perf_counter()
        classifier.predict(query)
        t1 = time.perf_counter()
        candidates = index.search(embedder.encode([query]), settings.retrieve_top_k)[0]
        t2 = time.perf_counter()
        reranker.rerank(query, candidates, settings.rerank_top_k)
        t3 = time.perf_counter()
        if record:
            stages["route"].append((t1 - t0) * 1000)
            stages["retrieve"].append((t2 - t1) * 1000)
            stages["rerank"].append((t3 - t2) * 1000)
            stages["total"].append((t3 - t0) * 1000)

    for query in queries:
        one(query, record=False)  # warm-up: pays model-load + first-call costs
    for _ in range(REPEATS):
        for query in queries:
            one(query, record=True)

    print(
        f"request-path latency (route -> retrieve -> rerank), "
        f"{len(stages['total'])} samples, CPU, generate excluded:"
    )
    for stage in ("route", "retrieve", "rerank", "total"):
        p50, p95, mean = _percentiles(stages[stage])
        print(f"  {stage:<9} p50={p50:6.1f}ms  p95={p95:6.1f}ms  mean={mean:6.1f}ms")
    print(f"  model={settings.embed_model}  retrieve_top_k={settings.retrieve_top_k}")


if __name__ == "__main__":
    main()
