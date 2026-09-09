"""
chunker.py — Role 2: Text Chunking with Overlap

Takes raw extracted text (from parser.py) and splits it into overlapping
chunks suitable for embedding + retrieval. Chunks are returned with basic
metadata so downstream modules (embedder.py, vector_store.py) can trace
each chunk back to its source.

No external dependencies — pure Python.
"""

from dataclasses import dataclass, field
from typing import List
import re
import uuid


@dataclass
class Chunk:
    """A single chunk of text plus metadata."""
    id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    source: str = "unknown"
    metadata: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    """Basic whitespace/newline normalization before chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)      # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse 3+ blank lines to 1
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
    source: str = "unknown",
) -> List[Chunk]:
    """
    Split text into overlapping chunks by character count, but snap chunk
    boundaries to the nearest sentence/paragraph break where possible so
    chunks stay semantically coherent instead of cutting mid-sentence.

    Args:
        text: Cleaned syllabus text.
        chunk_size: Target max characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.
        source: Identifier for the source document (e.g. filename).

    Returns:
        List of Chunk objects, in order, covering the full text.
    """
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = clean_text(text)
    chunks: List[Chunk] = []
    start = 0
    text_len = len(text)
    chunk_index = 0

    # Prefer breaking on paragraph, then sentence, then word boundaries.
    break_patterns = ["\n\n", ". ", "\n", " "]

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            window = text[start:end]
            snapped = False
            for pat in break_patterns:
                idx = window.rfind(pat)
                # Only snap if it doesn't shrink the chunk too aggressively
                if idx != -1 and idx > chunk_size * 0.4:
                    end = start + idx + len(pat)
                    snapped = True
                    break
            if not snapped:
                pass  # fall back to hard cut at chunk_size

        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_str,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    source=source,
                    metadata={"char_length": len(chunk_str)},
                )
            )
            chunk_index += 1

        if end >= text_len:
            break

        # Move start forward, backing up by `overlap` for context continuity
        start = max(end - overlap, start + 1)

    return chunks


def chunks_to_dicts(chunks: List[Chunk]) -> List[dict]:
    """Convenience: convert Chunk objects to plain dicts (e.g. for JSON/API)."""
    return [
        {
            "id": c.id,
            "text": c.text,
            "chunk_index": c.chunk_index,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "source": c.source,
            "metadata": c.metadata,
        }
        for c in chunks
    ]


if __name__ == "__main__":
    # Quick manual test — run `python chunker.py` to sanity check output.
    sample = """
    COURSE SYLLABUS: Introduction to Machine Learning

    Instructor: Dr. Jane Smith
    Office Hours: Tuesdays 2-4pm, Room 301

    Course Description:
    This course covers the fundamentals of machine learning, including
    supervised and unsupervised learning, neural networks, and model
    evaluation. Students will complete weekly homework assignments and
    a final project.

    Grading:
    Homework: 30%
    Midterm Exam: 25%
    Final Project: 30%
    Participation: 15%

    Late Policy:
    Assignments submitted late will lose 10% per day, up to 3 days late.
    After 3 days, no credit will be given without prior approval.
    """

    result = chunk_text(sample, chunk_size=300, overlap=50, source="ml_syllabus.pdf")
    print(f"Generated {len(result)} chunks:\n")
    for c in result:
        print(f"--- Chunk {c.chunk_index} ({c.start_char}-{c.end_char}, "
              f"{c.metadata['char_length']} chars) ---")
        print(c.text)
        print()