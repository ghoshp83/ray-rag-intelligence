"""Embedding entry guard. The fan-out machinery needs a Ray cluster, but the
empty-input check fires before any of it — and it must, because an empty corpus
is a setup error (no documents found), not a valid zero-vector index. Pinning it
keeps the failure loud (Rule 10) instead of letting an empty index build and a
silent zero-recall eval downstream.
"""

import pytest

from ray_rag.data.embed import embed_chunks


def test_embed_chunks_empty_input_fails_loud():
    # Runs without a Ray cluster: the guard precedes from_items/map_batches, so an
    # empty corpus raises here rather than building a degenerate index.
    with pytest.raises(ValueError, match="no chunks"):
        embed_chunks([], "BAAI/bge-small-en-v1.5")
