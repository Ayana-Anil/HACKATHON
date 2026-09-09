"""
chunker.py — Role 2: Text Chunking with Overlap

Responsibility: take a ParsedDocument (from Role 1's parser.py) and split
it into overlapping chunks suitable for embedding + retrieval.

Chunking happens PER PAGE rather than on the joined full_text. This is
deliberate: Role 5 (retriever.py) / Role 6 (llm.py) need to cite "which
page the answer came from" per PRD FR #7, so every chunk must carry an
accurate page_number. Chunking per page keeps that attribution exact —
chunking across page boundaries would blur which page a given chunk
actually came from.

Interface:
  - Input:  ParsedDocument (Role 1)
  - Output: list[Chunk]     (consumed by Role 3's embedder.py)

No external dependencies — pure Python.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List

try:
    # Normal case: running inside backend/ alongside parser.py
    from parser import ParsedDocument, Page
except ImportError:
    # Fallback so this file can still be imported/tested standalone
    # (e.g. before parser.py exists in this checkout, or in isolated unit tests).
    @dataclass
    class Page:  # type: ignore[no-redef]
        page_number: int
        text: str

    @dataclass
    class ParsedDocument:  # type: ignore[no-redef]
        filename: str
        pages: List[Page]
        full_text: str


@dataclass
class Chunk:
    """A single chunk of text plus metadata, ready for embedding."""
    id: str
    text: str
    chunk_index: int          # position of this chunk within the whole document
    page_number: int          # 1-indexed page this chunk came from (for citations)
    page_chunk_index: int     # position of this chunk within its page (0, 1, 2, ...)
    start_char: int           # char offset within the page's text
    end_char: int
    source: str = "unknown"   # filename, from ParsedDocument.filename
    metadata: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    """
    Extra whitespace normalization pass. parser.py already strips/joins lines,
    so this mainly guards against double spaces and stray control characters
    if chunker.py is ever fed text from another source.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_page_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[tuple[str, int, int]]:
    """
    Split a single page's text into (chunk_text, start_char, end_char) tuples.
    Snaps boundaries to paragraph -> sentence -> word breaks where possible
    so chunks don't cut mid-sentence.
    """
    text = clean_text(text)
    if not text:
        return []

    results: List[tuple[str, int, int]] = []
    start = 0
    text_len = len(text)
    break_patterns = ["\n\n", ". ", "\n", " "]

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            window = text[start:end]
            for pat in break_patterns:
                idx = window.rfind(pat)
                if idx != -1 and idx > chunk_size * 0.4:
                    end = start + idx + len(pat)
                    break
            # else: no good break found, hard cut at chunk_size

        piece = text[start:end].strip()
        if piece:
            results.append((piece, start, end))

        if end >= text_len:
            break

        start = max(end - overlap, start + 1)

    return results


def chunk_document(
    parsed: "ParsedDocument",
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Chunk a full parsed document, page by page, preserving page numbers.

    Args:
        parsed: ParsedDocument produced by parser.parse_pdf().
        chunk_size: Target max characters per chunk.
        overlap: Character overlap between consecutive chunks *within the
                  same page*. Overlap does not carry across page boundaries.

    Returns:
        List of Chunk objects in document order (page order, then position
        within page), each tagged with page_number for citation purposes.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    global_index = 0

    for page in parsed.pages:
        page_pieces = _chunk_page_text(page.text, chunk_size, overlap)
        for page_chunk_idx, (piece, start_char, end_char) in enumerate(page_pieces):
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=piece,
                    chunk_index=global_index,
                    page_number=page.page_number,
                    page_chunk_index=page_chunk_idx,
                    start_char=start_char,
                    end_char=end_char,
                    source=parsed.filename,
                    metadata={"char_length": len(piece)},
                )
            )
            global_index += 1

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
    source: str = "unknown",
    page_number: int = 1,
) -> List[Chunk]:
    """
    Lower-level helper: chunk a single raw string directly, without going
    through a ParsedDocument. Useful for unit tests or non-PDF text.
    All resulting chunks are tagged with the same page_number.
    """
    pieces = _chunk_page_text(text, chunk_size, overlap)
    chunks = []
    for i, (piece, start_char, end_char) in enumerate(pieces):
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=piece,
                chunk_index=i,
                page_number=page_number,
                page_chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                source=source,
                metadata={"char_length": len(piece)},
            )
        )
    return chunks


def chunks_to_dicts(chunks: List[Chunk]) -> List[dict]:
    """Convenience: convert Chunk objects to plain dicts (e.g. for JSON/API)."""
    return [
        {
            "id": c.id,
            "text": c.text,
            "chunk_index": c.chunk_index,
            "page_number": c.page_number,
            "page_chunk_index": c.page_chunk_index,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "source": c.source,
            "metadata": c.metadata,
        }
        for c in chunks
    ]


if __name__ == "__main__":
    # Quick manual test — run `python chunker.py` to sanity check output
    # using a fake 2-page ParsedDocument (no real PDF needed).
    fake_doc = ParsedDocument(
        filename="ml_syllabus.pdf",
        pages=[
            Page(
                page_number=1,
                text=(
                    "COURSE SYLLABUS: Introduction to Machine Learning\n"
                    "Instructor: Dr. Jane Smith\n"
                    "Office Hours: Tuesdays 2-4pm, Room 301\n"
                    "Course Description: This course covers the fundamentals of "
                    "machine learning, including supervised and unsupervised "
                    "learning, neural networks, and model evaluation. Students "
                    "will complete weekly homework assignments and a final project."
                ),
            ),
            Page(
                page_number=2,
                text=(
                    "Grading:\nHomework: 30%\nMidterm Exam: 25%\n"
                    "Final Project: 30%\nParticipation: 15%\n\n"
                    "Late Policy: Assignments submitted late will lose 10% per "
                    "day, up to 3 days late. After 3 days, no credit will be "
                    "given without prior approval."
                ),
            ),
        ],
        full_text="",  # not used by chunk_document
    )

    result = chunk_document(fake_doc, chunk_size=250, overlap=40)
    print(f"Generated {len(result)} chunks from {len(fake_doc.pages)} pages:\n")
    for c in result:
        print(f"--- Chunk {c.chunk_index} | page {c.page_number}, "
              f"chunk {c.page_chunk_index} on that page "
              f"({c.metadata['char_length']} chars) ---")
        print(c.text)
        print()
