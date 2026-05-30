"""Chunking correctness — boundaries and ids control what every downstream model sees."""

import pytest

from ray_rag.data.chunk import build_chunks, chunk_text


def test_chunk_text_windows_advance_and_cover():
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    # Window must advance (step = 30), so we expect coverage without runaway count.
    assert chunks[0].split()[0] == "w0"
    assert chunks[1].split()[0] == "w30"  # advanced by chunk_size - overlap
    assert chunks[-1].split()[-1] == "w99"  # last word is covered


def test_chunk_text_rejects_overlap_ge_size():
    # overlap >= size would never advance — must fail loud, not loop forever.
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=5, overlap=5)


def test_build_chunks_ids_are_unique_and_stable(tmp_path):
    (tmp_path / "doc.md").write_text(" ".join(f"w{i}" for i in range(80)))
    first = build_chunks(tmp_path, chunk_size=30, overlap=5)
    second = build_chunks(tmp_path, chunk_size=30, overlap=5)
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids))  # unique within corpus
    assert ids == [c.chunk_id for c in second]  # deterministic across runs
