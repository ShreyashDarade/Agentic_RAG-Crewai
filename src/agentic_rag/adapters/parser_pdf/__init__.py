"""PDF text extraction with PyMuPDF. Image-only (scanned) PDFs raise ``DocumentEmpty`` until OCR is added."""

from __future__ import annotations

import pymupdf

from agentic_rag.errors import DocumentEmpty, DocumentParseFailed, LimitExceeded
from agentic_rag.ports import ParsedDocument

__all__ = ["PdfParser"]


class PdfParser:
    extensions = frozenset({".pdf"})

    def __init__(self, *, max_pages: int = 500) -> None:
        self._max_pages = max_pages

    def parse(self, data: bytes, *, name: str) -> ParsedDocument:
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as exc:
            raise DocumentParseFailed() from exc
        with doc:
            if doc.needs_pass:
                raise DocumentParseFailed("the PDF is password protected")
            if doc.page_count > self._max_pages:
                raise LimitExceeded(details={"max_pages": self._max_pages})
            try:
                pages = tuple(doc.load_page(i).get_text("text").strip() for i in range(doc.page_count))
            except RuntimeError as exc:
                raise DocumentParseFailed() from exc
        if not any(pages):
            raise DocumentEmpty("the PDF has no extractable text (it may be scanned)")
        return ParsedDocument(text="\n\n".join(p for p in pages if p), pages=pages)
