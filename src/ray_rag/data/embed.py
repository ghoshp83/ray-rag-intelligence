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

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

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
    ds = ray.data.from_items([c.as_dict() for c in chunks])
    ds = ds.map_batches(
        Embedder,
        fn_constructor_kwargs={"model_name": model_name},
        batch_size=batch_size,
        concurrency=concurrency,
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

    if not ray.is_initialized():
        ray.init()
    index = build_corpus_index(settings.corpus_path, settings.embed_model, settings.index_path)
    print(f"built index: {len(index)} chunks -> {settings.index_path}")


if __name__ == "__main__":
    main()
