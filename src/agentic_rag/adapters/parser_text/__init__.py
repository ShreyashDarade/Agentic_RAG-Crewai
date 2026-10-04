"""Plain-text family: .txt .md .rst .csv (UTF-8 only; anything else is a typed failure, never mojibake)."""

from __future__ import annotations

from agentic_rag.errors import DocumentEmpty, DocumentParseFailed
from agentic_rag.ports import ParsedDocument

__all__ = ["TextParser"]


class TextParser:
    extensions = frozenset({".txt", ".md", ".rst", ".csv"})

    def parse(self, data: bytes, *, name: str) -> ParsedDocument:
        if b"\x00" in data:
            raise DocumentParseFailed("the file is binary, not text")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentParseFailed("the file is not valid UTF-8") from exc
        if not text.strip():
            raise DocumentEmpty()
        return ParsedDocument(text=text)
