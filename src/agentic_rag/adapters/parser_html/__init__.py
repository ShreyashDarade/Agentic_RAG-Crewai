"""HTML to text with BeautifulSoup. ``bs4`` and ``lxml`` are imported only here."""

from __future__ import annotations

from bs4 import BeautifulSoup

from agentic_rag.errors import DocumentEmpty, DocumentParseFailed
from agentic_rag.ports import ParsedDocument

__all__ = ["HtmlParser"]


class HtmlParser:
    extensions = frozenset({".html", ".htm"})
    version = "1"

    def parse(self, data: bytes, *, name: str) -> ParsedDocument:
        if b"\x00" in data:
            raise DocumentParseFailed("the file is binary, not HTML")
        try:
            soup = BeautifulSoup(data, "lxml")
        except Exception as exc:
            raise DocumentParseFailed() from exc
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()
        lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
        text = "\n".join(line for line in lines if line)
        if not text:
            raise DocumentEmpty()
        return ParsedDocument(text=text)
