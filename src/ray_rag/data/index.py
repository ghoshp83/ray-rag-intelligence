"""FAISS vector index + its chunk sidecar, kept together so they can't drift.

The index stores vectors; the sidecar JSONL stores the chunk metadata at the
same row positions. They are saved and loaded as a pair — a count mismatch on
load fails loud, because a desynced index would return chunk text that doesn't
match the vector that was actually matched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sidecar(index_path: str | Path) -> Path:
    return Path(index_path).with_suffix(".chunks.jsonl")


class VectorIndex:
    def __init__(self, faiss_index, chunks: list[dict[str, str]]):
        self._index = faiss_index
        self._chunks = chunks

    def __len__(self) -> int:
        return len(self._chunks)

    @classmethod
    def build(cls, embeddings: np.ndarray, chunks: list[dict[str, str]]) -> VectorIndex:
        import faiss

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"embeddings ({embeddings.shape[0]}) and chunks ({len(chunks)}) misaligned"
            )
        index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine via normalised vectors
        index.add(embeddings)
        return cls(index, chunks)

    def save(self, index_path: str | Path) -> None:
        import faiss

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        with _sidecar(index_path).open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(json.dumps(chunk) + "\n")

    @classmethod
    def load(cls, index_path: str | Path) -> VectorIndex:
        import faiss

        index = faiss.read_index(str(index_path))
        with _sidecar(index_path).open(encoding="utf-8") as fh:
            chunks = [json.loads(line) for line in fh]
        if index.ntotal != len(chunks):
            raise ValueError(
                f"index/sidecar desync: {index.ntotal} vectors vs {len(chunks)} chunks"
            )
        return cls(index, chunks)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> list[list[dict[str, object]]]:
        """Per query, return up to top_k chunk dicts with their similarity score."""
        scores, ids = self._index.search(query_embeddings.astype(np.float32), top_k)
        results: list[list[dict[str, object]]] = []
        for row_scores, row_ids in zip(scores, ids, strict=True):
            hits = [
                {**self._chunks[i], "score": float(s)}
                for s, i in zip(row_scores, row_ids, strict=True)
                if i != -1
            ]
            results.append(hits)
        return results
