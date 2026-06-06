"""Measure Ray Data embedding throughput — the parallel batch-inference showcase.

Times the `map_batches` embedding pass and reports chunks/sec alongside the batch
size, actor concurrency, and the Ray cluster CPU count, so the figure is read in
the context of the resources that produced it (it scales with cluster CPUs).

The bundled corpus is tiny (~20 chunks), so a raw pass would measure one-off
actor/model startup, not embedding speed. To report *steady-state* throughput we
replicate the corpus chunks up to a target count, then run one warm-up pass
(discarded — it pays the model-load cost) before the timed pass. CPU-only, no
GPU, no API key.

Run: `python scripts/measure_throughput.py` (or `make bench`).
"""

from __future__ import annotations

import time
from dataclasses import replace

import ray

from ray_rag.config import settings
from ray_rag.data.chunk import build_chunks
from ray_rag.data.embed import embed_chunks

TARGET_CHUNKS = 2000  # replicate corpus to this size so startup amortises
BATCH_SIZE = 128  # larger batches amortise per-batch overhead on CPU
CONCURRENCY = 4


def main() -> None:
    if not ray.is_initialized():
        ray.init()

    base = build_chunks(settings.corpus_path)
    if not base:
        raise ValueError("no chunks built — is the corpus empty?")
    # Replicate to TARGET_CHUNKS with unique ids so rows stay distinct.
    chunks = [replace(base[i % len(base)], chunk_id=f"bench-{i}") for i in range(TARGET_CHUNKS)]

    embed = lambda: embed_chunks(  # noqa: E731 - tiny local timing closure
        chunks, settings.embed_model, batch_size=BATCH_SIZE, concurrency=CONCURRENCY
    )
    embed()  # warm-up: pays the model-load cost so the timed pass is steady state

    start = time.perf_counter()
    embeddings, _ = embed()
    elapsed = time.perf_counter() - start

    cpus = ray.cluster_resources().get("CPU", 0)
    print(
        f"embedding throughput: {len(embeddings) / elapsed:.0f} chunks/sec "
        f"({len(embeddings)} chunks in {elapsed:.2f}s)"
    )
    print(
        f"  model={settings.embed_model}  batch_size={BATCH_SIZE}  "
        f"concurrency={CONCURRENCY}  cluster_CPUs={cpus:.0f}"
    )


if __name__ == "__main__":
    main()
