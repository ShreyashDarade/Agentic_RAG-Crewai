from __future__ import annotations

from typing import Any

from agentic_rag.adapters.chunker_recursive import RecursiveChunker
from agentic_rag.adapters.lexical_bm25 import Bm25Index
from agentic_rag.adapters.parser_text import TextParser
from agentic_rag.testing import (
    AnswerPipelineContract,
    ChatModelContract,
    ChunkerContract,
    EmbedderContract,
    FakeChatModel,
    FakeEmbedder,
    LexicalIndexContract,
    ParserContract,
)


class TestBm25Index(LexicalIndexContract):
    def create(self) -> Any:
        return Bm25Index()


class TestRecursiveChunker(ChunkerContract):
    def create(self) -> Any:
        return RecursiveChunker(max_chars=400, overlap_chars=60)


class TestTextParser(ParserContract):
    def create(self) -> Any:
        return TextParser()

    def sample(self) -> bytes:
        return b"hello world"

    def corrupt(self) -> bytes:
        return b"\xff\xfe\x00\x01 not text"

    def empty(self) -> bytes:
        return b"  \n\t "


class TestFakeEmbedder(EmbedderContract):
    async def create(self) -> Any:
        return FakeEmbedder(16)

    async def create_failing(self) -> Any:
        return FakeEmbedder(16, fail=True)


class TestFakeChatModel(ChatModelContract):
    async def create(self) -> Any:
        return FakeChatModel(["ok"])

    async def create_failing(self) -> Any:
        return FakeChatModel(fail=True)


class TestHtmlParser(ParserContract):
    sample_name = "page.html"
    sample_expect = "Hello"

    def create(self) -> Any:
        from agentic_rag.adapters.parser_html import HtmlParser

        return HtmlParser()

    def sample(self) -> bytes:
        return b"<html><head><script>x=1</script></head><body><h1>Hello</h1><p>world</p></body></html>"

    def corrupt(self) -> bytes:
        return b"\x00\x01\x02 binary"

    def empty(self) -> bytes:
        return b"<html><body><script>only()</script></body></html>"


class TestPdfParser(ParserContract):
    sample_name = "a.pdf"
    sample_expect = "Hello"

    def create(self) -> Any:
        from agentic_rag.adapters.parser_pdf import PdfParser

        return PdfParser()

    def sample(self) -> bytes:
        import pymupdf

        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "Hello PDF world")
        data: bytes = doc.tobytes()
        return data

    def corrupt(self) -> bytes:
        return b"%PDF-1.7 garbage that is not a pdf"

    def empty(self) -> bytes:
        import pymupdf

        doc = pymupdf.open()
        doc.new_page()
        data: bytes = doc.tobytes()
        return data


class TestDocxParser(ParserContract):
    sample_name = "a.docx"
    sample_expect = "Hello"

    def create(self) -> Any:
        from agentic_rag.adapters.parser_docx import DocxParser

        return DocxParser()

    def _build(self, text: str | None) -> bytes:
        import io

        import docx

        document = docx.Document()
        if text:
            document.add_paragraph(text)
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()

    def sample(self) -> bytes:
        return self._build("Hello docx world")

    def corrupt(self) -> bytes:
        return b"PK\x03\x04 not really a zip"

    def empty(self) -> bytes:
        return self._build(None)


class TestDirectPipeline(AnswerPipelineContract):
    def create(self, chat: Any) -> Any:
        from agentic_rag.adapters.pipeline_direct import DirectPipeline

        return DirectPipeline(chat)
