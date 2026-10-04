"""Fakes and conformance suites for plug-in authors (stable tier).

Standard library plus the ports only: importable on a thin install.
"""

from agentic_rag.testing.contracts import (
    ChatModelContract,
    ChunkerContract,
    EmbedderContract,
    LexicalIndexContract,
    ParserContract,
    RerankerContract,
    VectorStoreContract,
    make_chunks,
)
from agentic_rag.testing.fakes import FakeChatModel, FakeEmbedder, FakeVectorStore

__all__ = [
    "ChatModelContract",
    "ChunkerContract",
    "EmbedderContract",
    "FakeChatModel",
    "FakeEmbedder",
    "FakeVectorStore",
    "LexicalIndexContract",
    "ParserContract",
    "RerankerContract",
    "VectorStoreContract",
    "make_chunks",
]
