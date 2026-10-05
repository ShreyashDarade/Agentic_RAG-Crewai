"""DOCX text extraction with python-docx, with zip-bomb guards applied before any parsing."""

from __future__ import annotations

import io
import zipfile

import docx
from docx.table import Table

from agentic_rag.errors import DocumentEmpty, DocumentParseFailed, LimitExceeded
from agentic_rag.ports import ParsedDocument

__all__ = ["DocxParser"]


class DocxParser:
    extensions = frozenset({".docx"})
    version = "2"  # 2: document order, merged cells once

    def __init__(
        self,
        *,
        max_uncompressed_bytes: int = 32 * 1024 * 1024,
        max_member_bytes: int = 8 * 1024 * 1024,
        max_members: int = 5000,
    ) -> None:
        self._max_bytes = max_uncompressed_bytes
        self._max_member = max_member_bytes
        self._max_members = max_members

    def parse(self, data: bytes, *, name: str) -> ParsedDocument:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = archive.infolist()
        except zipfile.BadZipFile as exc:
            raise DocumentParseFailed("the file is not a valid DOCX") from exc
        # python-docx needs roughly 25x the XML size in memory and about a second per MiB, so the bounds are on the
        # declared size of each part (``word/document.xml`` is the one that grows) as well as on the total.
        if (
            len(members) > self._max_members
            or sum(m.file_size for m in members) > self._max_bytes
            or any(m.file_size > self._max_member for m in members)
        ):
            raise LimitExceeded(details={"max_uncompressed_bytes": self._max_bytes})
        try:
            document = docx.Document(io.BytesIO(data))
        except Exception as exc:
            raise DocumentParseFailed() from exc
        parts: list[str] = []
        for block in document.iter_inner_content():  # paragraphs and tables in the order they appear
            if isinstance(block, Table):
                parts.extend(_table_rows(block))
            else:
                parts.append(block.text.strip())
        text = "\n".join(p for p in parts if p)
        if not text:
            raise DocumentEmpty()
        return ParsedDocument(text=text)


def _table_rows(table: Table) -> list[str]:
    rows: list[str] = []
    for row in table.rows:
        seen: set[int] = set()
        cells: list[str] = []
        for cell in row.cells:  # a merged cell is returned once per grid column it spans
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            if cell.text.strip():
                cells.append(cell.text.strip())
        if cells:
            rows.append(" | ".join(cells))
    return rows
