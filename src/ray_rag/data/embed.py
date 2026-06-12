"""Distributed embedding with Ray Data — the parallel batch-inference showcase.

The embedding model is loaded once per worker (stateful `map_batches` actor),
not once per row, so cost is amortised across the corpus and scales with cluster
CPUs. The same `Embedder` is reused at serve time to embed queries, guaranteeing
query and corpus vectors come from the identical model (a mismatch would
silently wreck retrieval — so we share one class).
"""

from __future__ import annotations

import numpy as np
import ray

from ray_rag.data.chunk import Chunk, build_chunks
from ray_rag.data.index import VectorIndex


class Embedder:
    """Encodes text to L2-normalised vectors (so inner product == cosine)."""

    def __init__(self, model_name: str, num_threads: int | None = None):
        from sentence_transformers import SentenceTransformer

        # Bound this actor's torch threads so a pool of actors fans out across
        # CPUs instead of each one grabbing every core and thrashing. Only set
        # inside the Ray actor pool (where num_threads is passed); left default
        # for direct in-process use (serve/eval query embedding).
        if num_threads is not None:
            import torch

            torch.set_num_threads(num_threads)
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(
            np.float32
        )

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        batch["embedding"] = self.encode(list(batch["text"]))
        return batch


def embed_chunks(
    chunks: list[Chunk],
    model_name: str,
    batch_size: int = 32,
    concurrency: int = 4,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Run chunks through Ray Data; return (embeddings, chunk-metadata) aligned by row."""
    if not chunks:
        raise ValueError("no chunks to embed — is the corpus empty?")
    # from_items packs everything into one block by default, which pins the
    # whole pass to a single actor (1 CPU) however high `concurrency` is. Split
    # into at least `concurrency` blocks so the actor pool actually fans out
    # across cluster CPUs — that fan-out is the parallel-batch-inference showcase.
    blocks = min(concurrency, len(chunks))
    # Give each actor a slice of the cores (threads × concurrency ≈ cluster CPUs)
    # and reserve that many CPUs per actor, so the pool parallelises instead of
    # every actor's torch grabbing all cores and thrashing (oversubscription).
    total_cpus = int(ray.cluster_resources().get("CPU", concurrency))
    threads = max(1, total_cpus // concurrency)
    ds = ray.data.from_items([c.as_dict() for c in chunks], override_num_blocks=blocks)
    ds = ds.map_batches(
        Embedder,
        fn_constructor_kwargs={"model_name": model_name, "num_threads": threads},
        batch_size=batch_size,
        concurrency=concurrency,
        num_cpus=threads,
    )
    rows = ds.take_all()
    embeddings = np.vstack([r.pop("embedding") for r in rows]).astype(np.float32)
    return embeddings, rows


def build_corpus_index(
    corpus_path: str,
    model_name: str,
    index_path: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> VectorIndex:
    """End-to-end: corpus -> chunks -> embeddings -> saved FAISS index."""
    chunks = build_chunks(corpus_path, chunk_size, overlap)
    embeddings, meta = embed_chunks(chunks, model_name)
    index = VectorIndex.build(embeddings, meta)
    index.save(index_path)
    return index


def main() -> None:
    from ray_rag.config import settings
    from ray_rag.observability import log_event

    if not ray.is_initialized():
        ray.init()
    index = build_corpus_index(settings.corpus_path, settings.embed_model, settings.index_path)
    print(f"built index: {len(index)} chunks -> {settings.index_path}")
    # The observability schema advertises an `ingest` component; emit the build as
    # one structured event so the index-build step is greppable like the rest.
    log_event("ingest", "index_built", n_chunks=len(index), index_path=settings.index_path)


if __name__ == "__main__":
    main()
