"""Retrieval-index correctness: the vector→chunk row alignment is the foundation
the whole RAG pipeline stands on. If `search` returns a chunk that doesn't match
the vector it actually matched, every downstream stage (rerank, generate, cite)
grounds on the wrong text — silently. These tests pin the alignment, the top_k
cap, the -1 padding filter, and the fail-loud desync guards.
"""

import numpy as np
import pytest

from ray_rag.data.index import VectorIndex

# Orthogonal unit vectors so inner-product (cosine) ranking is unambiguous: a
# query aligned with one axis must return that chunk first, score ~1.0.
_EMB = np.eye(3, dtype=np.float32)
_CHUNKS = [
    {"chunk_id": "a.md#0-aaaaaaaaaaaa", "text": "alpha", "source": "a.md"},
    {"chunk_id": "b.md#1-bbbbbbbbbbbb", "text": "bravo", "source": "b.md"},
    {"chunk_id": "c.md#2-cccccccccccc", "text": "charlie", "source": "c.md"},
]


def test_search_returns_aligned_chunk_with_score():
    index = VectorIndex.build(_EMB, _CHUNKS)
    [hits] = index.search(np.array([[0, 1, 0]], dtype=np.float32), top_k=1)
    assert hits[0]["chunk_id"] == "b.md#1-bbbbbbbbbbbb"  # the matched vector's chunk
    assert hits[0]["text"] == "bravo"
    assert hits[0]["score"] == pytest.approx(1.0)


def test_search_caps_at_top_k_and_orders_by_similarity():
    index = VectorIndex.build(_EMB, _CHUNKS)
    # Query closest to axis 0, then axis 2 (axis 1 is orthogonal -> 0).
    query = np.array([[0.9, 0.0, 0.1]], dtype=np.float32)
    [hits] = index.search(query, top_k=2)
    assert len(hits) == 2  # capped at top_k even though 3 vectors exist
    assert [h["chunk_id"] for h in hits] == [
        "a.md#0-aaaaaaaaaaaa",
        "c.md#2-cccccccccccc",
    ]
    assert hits[0]["score"] >= hits[1]["score"]


def test_search_drops_minus_one_padding_when_top_k_exceeds_corpus():
    # faiss pads with -1 when fewer than top_k vectors exist; those must be
    # filtered, not indexed as chunk[-1] (which would return the last chunk).
    index = VectorIndex.build(_EMB, _CHUNKS)
    [hits] = index.search(np.array([[1, 0, 0]], dtype=np.float32), top_k=5)
    assert len(hits) == 3


def test_build_rejects_misaligned_embeddings_and_chunks():
    with pytest.raises(ValueError):
        VectorIndex.build(_EMB[:2], _CHUNKS)  # 2 vectors, 3 chunks


def test_save_load_round_trips(tmp_path):
    index_path = tmp_path / "idx.faiss"
    VectorIndex.build(_EMB, _CHUNKS).save(index_path)
    loaded = VectorIndex.load(index_path)
    assert len(loaded) == 3
    [hits] = loaded.search(np.array([[0, 0, 1]], dtype=np.float32), top_k=1)
    assert hits[0]["chunk_id"] == "c.md#2-cccccccccccc"


def test_load_fails_loud_on_index_sidecar_desync(tmp_path):
    # A sidecar with more rows than the index means metadata no longer lines up
    # with vectors — loading must raise, not silently return mismatched chunks.
    index_path = tmp_path / "idx.faiss"
    VectorIndex.build(_EMB, _CHUNKS).save(index_path)
    sidecar = index_path.with_suffix(".chunks.jsonl")
    with sidecar.open("a", encoding="utf-8") as fh:
        fh.write('{"chunk_id": "d.md#3-dddddddddddd", "text": "delta", "source": "d.md"}\n')
    with pytest.raises(ValueError):
        VectorIndex.load(index_path)
