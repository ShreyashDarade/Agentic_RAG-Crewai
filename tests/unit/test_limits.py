"""G13: every bound on attacker- or accident-controlled input has a test that exceeds it and sees the typed rejection."""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Sequence
from typing import Any

import pytest

from agentic_rag.adapters.parser_text import TextParser
from agentic_rag.application import Limits
from agentic_rag.contracts import QueryRequest, SearchRequest
from agentic_rag.errors import (
    DeadlineExceeded,
    LimitExceeded,
    PayloadTooLarge,
    ValidationFailed,
)
from agentic_rag.ports import ParsedDocument
from agentic_rag.testing import FakeEmbedder
from tests.unit.test_service import TEXT, build


async def test_max_chunks_per_document() -> None:
    service, *_ = build(limits=Limits(max_chunks_per_document=2))
    with pytest.raises(LimitExceeded) as err:
        await service.ingest_document("a.txt", TEXT)  # three chunks of at most 100 characters
    assert err.value.details == {"max_chunks_per_document": 2}


@pytest.mark.parametrize("limit", [0, -1, 101, 10_000])
async def test_max_page_size(limit: int) -> None:
    service, *_ = build()
    with pytest.raises(ValidationFailed):
        await service.list_documents(limit=limit)


async def test_max_upload_bytes_question_chars_and_top_k_over_the_service() -> None:
    service, *_ = build(limits=Limits(max_upload_bytes=10, max_question_chars=5, max_top_k=2))
    with pytest.raises(PayloadTooLarge):
        await service.ingest_document("a.txt", b"x" * 11)
    with pytest.raises(LimitExceeded):
        await service.search(SearchRequest(query="123456"))
    with pytest.raises(LimitExceeded):
        await service.query(QueryRequest(question="123456"))
    with pytest.raises(LimitExceeded):
        await service.search(SearchRequest(query="ok", top_k=3))


async def test_embed_batch_size_splits_the_calls() -> None:
    embedder = FakeEmbedder(32)
    service, *_ = build(embedder=embedder, limits=Limits(embed_batch_size=2))
    result = await service.ingest_document("a.txt", TEXT)
    assert embedder.calls == math.ceil(result.chunks_indexed / 2) > 1


class _Counting(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(32)
        self.running = 0
        self.peak = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(0.05)
            return await super().embed_documents(texts)
        finally:
            self.running -= 1


async def test_max_concurrent_embed_batches() -> None:
    embedder = _Counting()
    service, *_ = build(embedder=embedder, limits=Limits(embed_batch_size=1, max_concurrent_embed_batches=2))
    await service.ingest_document("a.txt", TEXT * 2)
    assert embedder.peak == 2


class _Gate(TextParser):
    def __init__(self) -> None:
        self.running = 0
        self.peak = 0
        self._lock = threading.Lock()

    def parse(self, data: bytes, *, name: str) -> ParsedDocument:
        with self._lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            threading.Event().wait(0.1)
            return super().parse(data, name=name)
        finally:
            with self._lock:
                self.running -= 1


async def test_max_concurrent_ingests() -> None:
    parser = _Gate()
    service, *_ = build(limits=Limits(max_concurrent_ingests=2))
    service._parsers[".txt"] = parser
    await asyncio.gather(*(service.ingest_document("a.txt", TEXT + str(i).encode()) for i in range(6)))
    assert parser.peak == 2


async def test_request_deadline_and_health_check_timeout() -> None:
    class Hangs:
        async def check(self) -> None:
            await asyncio.sleep(30)

    service, store, _ = build(limits=Limits(request_deadline_seconds=0.2, health_check_timeout_seconds=0.2))
    service._health["vector_store"] = Hangs()  # type: ignore[assignment]
    report = await service.ready()
    assert report.checks["vector_store"] == "timeout"

    async def slow(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(5)

    service._retriever.retrieve = slow  # type: ignore[method-assign]
    with pytest.raises(DeadlineExceeded):
        await service.search(SearchRequest(query="apples"))


def test_pdf_max_pages() -> None:
    pymupdf = pytest.importorskip("pymupdf")
    from agentic_rag.adapters.parser_pdf import PdfParser

    doc = pymupdf.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"page {i} has some text")
    data = doc.tobytes()
    with pytest.raises(LimitExceeded) as err:
        PdfParser(max_pages=2).parse(data, name="a.pdf")
    assert err.value.details == {"max_pages": 2}
    assert PdfParser(max_pages=3).parse(data, name="a.pdf").pages


def test_docx_members_and_bytes() -> None:
    import io
    import zipfile

    pytest.importorskip("docx")
    from agentic_rag.adapters.parser_docx import DocxParser

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for i in range(5):
            archive.writestr(f"m{i}.xml", "x")
    with pytest.raises(LimitExceeded):
        DocxParser(max_members=4).parse(buffer.getvalue(), name="a.docx")
    with pytest.raises(LimitExceeded):
        DocxParser(max_uncompressed_bytes=3).parse(buffer.getvalue(), name="a.docx")
