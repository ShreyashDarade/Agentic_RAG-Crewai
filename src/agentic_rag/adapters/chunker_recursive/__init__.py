"""Boundary-aware character chunker: paragraphs, then sentences, then words; overlap between chunks."""

from __future__ import annotations

import hashlib
import re

from agentic_rag.errors import ConfigurationError
from agentic_rag.ports import Chunk, ParsedDocument

__all__ = ["RecursiveChunker"]

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class RecursiveChunker:
    """Chunks never span pages. Defaults are unmeasured (framework section 11)."""

    version = "recursive-1"

    def __init__(self, *, max_chars: int = 1500, overlap_chars: int = 200) -> None:
        if max_chars < 100 or not 0 <= overlap_chars < max_chars // 2:
            raise ConfigurationError("chunk size must be >= 100 and overlap < half of it")
        self._max = max_chars
        self._overlap = overlap_chars

    def chunk(self, doc: ParsedDocument, *, document_id: str, document_name: str) -> list[Chunk]:
        pages: list[tuple[int | None, str]]
        pages = [(i, text) for i, text in enumerate(doc.pages, start=1)] if doc.pages else [(None, doc.text)]
        chunks: list[Chunk] = []
        for page, text in pages:
            for piece in self._split(text):
                index = len(chunks)
                cid = (
                    "ch_" + hashlib.sha256(f"{document_id}:{index}:{self.version}".encode()).hexdigest()[:32]
                )
                chunks.append(
                    Chunk(
                        id=cid,
                        document_id=document_id,
                        index=index,
                        text=piece,
                        document_name=document_name,
                        page=page,
                    )
                )
        return chunks

    def _units(self, text: str) -> list[str]:
        units: list[str] = []
        for para in _PARAGRAPH.split(text):
            para = para.strip()
            if not para:
                continue
            if len(para) <= self._max:
                units.append(para)
                continue
            for sentence in _SENTENCE.split(para):
                units.extend(self._hard_split(sentence.strip()))
        return [u for u in units if u]

    def _hard_split(self, text: str) -> list[str]:
        out: list[str] = []
        while len(text) > self._max:
            cut = text.rfind(" ", 0, self._max)
            cut = cut if cut > 0 else self._max
            out.append(text[:cut].strip())
            text = text[cut:].strip()
        out.append(text)
        return out

    def _split(self, text: str) -> list[str]:
        pieces: list[str] = []
        current: list[str] = []
        size = 0
        for unit in self._units(text):
            if current and size + 1 + len(unit) > self._max:
                emitted = "\n".join(current)
                pieces.append(emitted)
                tail = self._tail(emitted)
                current, size = ([tail], len(tail)) if tail else ([], 0)
                if size and size + 1 + len(unit) > self._max:
                    current, size = [], 0  # the overlap would push this chunk over the limit
            current.append(unit)
            size += len(unit) + (1 if size else 0)
        if current and any(u.strip() for u in current):
            last = "\n".join(current)
            if not pieces or last != pieces[-1]:
                pieces.append(last)
        return pieces

    def _tail(self, text: str) -> str:
        if self._overlap == 0:
            return ""
        tail = text[-self._overlap :]
        if " " in tail and len(text) > self._overlap:
            tail = tail.split(" ", 1)[1]
        return tail.strip()
