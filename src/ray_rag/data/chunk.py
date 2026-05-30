"""Document loading and chunking — pure, deterministic, unit-testable.

Kept free of Ray and model dependencies so chunking logic can be tested in
isolation: a chunk-boundary regression is a correctness bug (it changes what the
reranker and LLM ever see), so it must fail a test, not slip through.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

_TEXT_SUFFIXES = {".txt", ".md"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def load_documents(corpus_path: str | Path) -> Iterator[tuple[str, str, str]]:
    """Yield (doc_id, source, text) for each text/markdown file under corpus_path."""
    root = Path(corpus_path)
    if not root.exists():
        raise FileNotFoundError(f"corpus path does not exist: {root}")
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in _TEXT_SUFFIXES and path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                doc_id = str(path.relative_to(root))
                yield doc_id, str(path), text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window split over whitespace tokens.

    chunk_size and overlap are in tokens; overlap must be < chunk_size so the
    window always advances (else it would loop forever — fail loud).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]
    # Drop a trailing chunk fully contained in the previous one (overlap artifact).
    if len(chunks) > 1 and len(words) <= (len(chunks) - 1) * step + overlap:
        chunks.pop()
    return chunks


def _chunk_id(doc_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}:{index}:{text}".encode()).hexdigest()[:12]
    return f"{doc_id}#{index}-{digest}"


def build_chunks(corpus_path: str | Path, chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Load every document and flatten into stable-id chunks."""
    chunks: list[Chunk] = []
    for doc_id, source, text in load_documents(corpus_path):
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(_chunk_id(doc_id, i, piece), doc_id, source, piece))
    return chunks
