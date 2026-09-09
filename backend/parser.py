"""
parser.py — Role 1: PDF text extraction

Responsibility: given a syllabus PDF, extract clean, page-level text
that downstream roles can consume:
  - Role 2 (chunker.py) chunks the returned text
  - Role 5 (retriever.py) / Role 6 (llm.py) use page numbers to cite
    "which section the answer was based on" (FR #7)
  - Role 7 (main.py) calls parse_pdf() directly from the /upload endpoint

Per PRD §8: parsing is pypdf-only, PDF-only for v1 (no DOCX, no OCR).
Per PRD §11 (Risks): must fail gracefully on unreadable/complex PDFs
rather than crashing the request.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


class PDFParsingError(Exception):
    """Raised when a PDF cannot be parsed or contains no extractable text."""


@dataclass
class Page:
    """One page of extracted text, 1-indexed to match how humans refer to pages."""
    page_number: int
    text: str


@dataclass
class ParsedDocument:
    """Full parsing result handed off to chunker.py / main.py."""
    filename: str
    pages: list[Page]
    full_text: str  # convenience: all pages joined, for simple chunking strategies

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def has_text(self) -> bool:
        return bool(self.full_text.strip())


def parse_pdf(source: Union[str, Path, io.BytesIO, bytes]) -> ParsedDocument:
    """
    Extract text from a PDF, page by page.

    Args:
        source: a file path (str/Path), raw bytes, or a file-like object
                 (e.g. the .file attribute of a FastAPI UploadFile).

    Returns:
        ParsedDocument with per-page text and a convenience full_text field.

    Raises:
        PDFParsingError: if the file can't be read as a PDF, or if it parses
                          but yields no extractable text (e.g. a scanned/
                          image-only PDF — OCR is out of scope for v1, so we
                          surface this as a clear error rather than silently
                          returning nothing).
    """
    filename = getattr(source, "name", None) or (
        str(source) if isinstance(source, (str, Path)) else "uploaded.pdf"
    )

    try:
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        reader = PdfReader(source)
    except PdfReadError as e:
        raise PDFParsingError(f"Could not read '{filename}' as a PDF: {e}") from e
    except FileNotFoundError as e:
        raise PDFParsingError(f"File not found: {filename}") from e
    except Exception as e:  # noqa: BLE001 — surface any pypdf failure as our own error type
        raise PDFParsingError(f"Unexpected error parsing '{filename}': {e}") from e

    if reader.is_encrypted:
        # Try an empty password first (common for "restricted but not really locked" PDFs)
        try:
            reader.decrypt("")
        except Exception as e:
            raise PDFParsingError(
                f"'{filename}' is password-protected and could not be decrypted."
            ) from e

    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001 — one bad page shouldn't kill the whole doc
            logger.warning("Failed to extract text from page %d of '%s': %s", i, filename, e)
            raw_text = ""
        pages.append(Page(page_number=i, text=_clean_text(raw_text)))

    full_text = "\n\n".join(p.text for p in pages if p.text)

    doc = ParsedDocument(filename=filename, pages=pages, full_text=full_text)

    if not doc.has_text:
        raise PDFParsingError(
            f"'{filename}' parsed successfully but contains no extractable text. "
            "This is likely a scanned/image-only PDF, which is out of scope for v1 "
            "(no OCR support)."
        )

    return doc


def _clean_text(text: str) -> str:
    """
    Light normalization: pypdf's extract_text() often produces irregular
    whitespace (extra newlines mid-sentence, double spaces from multi-column
    layouts, etc). Chunking/embedding work better on normalized text.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop empty lines
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual smoke test:
    #   python parser.py path/to/syllabus.pdf
    import sys

    if len(sys.argv) != 2:
        print("Usage: python parser.py <path_to_pdf>")
        sys.exit(1)

    try:
        result = parse_pdf(sys.argv[1])
        print(f"Parsed '{result.filename}': {result.num_pages} pages, "
              f"{len(result.full_text)} chars extracted.")
        print("\n--- First page preview ---")
        print(result.pages[0].text[:500])
    except PDFParsingError as e:
        print(f"Parsing failed: {e}")
        sys.exit(1)
