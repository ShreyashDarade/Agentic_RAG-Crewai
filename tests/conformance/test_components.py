from __future__ import annotations

from typing import Any

from agentic_rag.adapters.chunker_recursive import RecursiveChunker
from agentic_rag.adapters.lexical_bm25 import Bm25Index
from agentic_rag.adapters.parser_text import TextParser
from agentic_rag.testing import (
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
