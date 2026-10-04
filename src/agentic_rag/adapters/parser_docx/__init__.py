"""DOCX text extraction with python-docx, with zip-bomb guards applied before any parsing."""

from __future__ import annotations

import io
import zipfile

import docx

from agentic_rag.errors import DocumentEmpty, DocumentParseFailed, LimitExceeded
from agentic_rag.ports import ParsedDocument

__all__ = ["DocxParser"]


class DocxParser:
    extensions = frozenset({".docx"})

    def __init__(self, *, max_uncompressed_bytes: int = 100 * 1024 * 1024, max_members: int = 5000) -> None:
        self._max_bytes = max_uncompressed_bytes
        self._max_members = max_members

    def parse(self, data: bytes, *, name: str) -> ParsedDocument:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = archive.infolist()
        except zipfile.BadZipFile as exc:
            raise DocumentParseFailed("the file is not a valid DOCX") from exc
        if len(members) > self._max_members or sum(m.file_size for m in members) > self._max_bytes:
            raise LimitExceeded(details={"max_uncompressed_bytes": self._max_bytes})
        try:
            document = docx.Document(io.BytesIO(data))
        except Exception as exc:
            raise DocumentParseFailed() from exc
        parts = [p.text.strip() for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        text = "\n".join(p for p in parts if p)
        if not text:
            raise DocumentEmpty()
        return ParsedDocument(text=text)
