"""Chunking correctness — boundaries and ids control what every downstream model sees."""

import pytest

from ray_rag.data.chunk import build_chunks, chunk_text, load_documents


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


def test_chunk_text_drops_trailing_window_fully_inside_its_predecessor():
    # 70 words, size 40, overlap 10 (step 30): windows start at 0, 30, 60. The
    # third (w60..w69) is entirely contained in the second (w30..w69), so it is a
    # pure overlap artifact and must be dropped — re-emitting it would feed the
    # reranker and the LLM the same passage twice and inflate the chunk count.
    words = [f"w{i}" for i in range(70)]
    chunks = chunk_text(" ".join(words), chunk_size=40, overlap=10)
    assert len(chunks) == 2
    assert chunks[-1].split()[-1] == "w69"  # every word still covered
    # The surviving windows overlap by exactly `overlap` tokens — the context
    # continuity that overlap exists to preserve across a boundary.
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]


def test_chunk_text_keeps_trailing_window_that_carries_a_new_token():
    # One more word (71): the third window now ends at w70, a token no earlier
    # window holds, so it is NOT an artifact and must be kept — the drop fires only
    # when the tail is redundant, never when it carries unique content.
    words = [f"w{i}" for i in range(71)]
    chunks = chunk_text(" ".join(words), chunk_size=40, overlap=10)
    assert len(chunks) == 3
    assert chunks[-1].split()[-1] == "w70"


def test_load_documents_filters_to_nonempty_text_files_with_relative_ids(tmp_path):
    # The corpus gate decides what ever reaches the index. A non-text file must be
    # ignored (we would otherwise embed binary junk), a whitespace-only file dropped
    # (an empty chunk grounds nothing), and a nested doc keyed by its path relative
    # to the corpus root so chunk ids stay stable and human-readable.
    (tmp_path / "keep.md").write_text("hello world")
    (tmp_path / "blank.md").write_text("   \n  ")  # whitespace only -> skipped
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")  # non-text suffix -> skipped
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("  deep text  ")

    docs = list(load_documents(tmp_path))
    doc_ids = [doc_id for doc_id, _source, _text in docs]
    assert doc_ids == ["keep.md", "sub/nested.txt"]  # sorted, blank + png excluded
    texts = {doc_id: text for doc_id, _source, text in docs}
    assert texts["sub/nested.txt"] == "deep text"  # surrounding whitespace stripped


def test_load_documents_missing_corpus_fails_loud(tmp_path):
    # A missing corpus path is a setup error, not an empty corpus — it must raise,
    # not silently yield nothing and let a downstream "no chunks" error mislead.
    with pytest.raises(FileNotFoundError):
        list(load_documents(tmp_path / "does_not_exist"))


def test_build_chunks_ids_are_unique_and_stable(tmp_path):
    (tmp_path / "doc.md").write_text(" ".join(f"w{i}" for i in range(80)))
    first = build_chunks(tmp_path, chunk_size=30, overlap=5)
    second = build_chunks(tmp_path, chunk_size=30, overlap=5)
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids))  # unique within corpus
    assert ids == [c.chunk_id for c in second]  # deterministic across runs
