"""Ports: small protocols, one capability each (framework section 7, rule I).

No third-party imports. Adapters implement them; ``application`` depends only on them.
Every method documents its failure contract: it raises a :class:`agentic_rag.errors.RagError`
subclass, never a dependency's exception and never a silent default.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Collection, Sequence
from typing import Protocol, runtime_checkable

from agentic_rag.ports.types import (
    ChatMessage,
    Chunk,
    ChunkFilter,
    Completion,
    DocumentRecord,
    ParsedDocument,
    PipelineAnswer,
    ScoredChunk,
)

__all__ = [
    "AnswerPipeline",
    "ChatModel",
    "ChunkScanner",
    "Chunker",
    "DocumentCatalog",
    "DocumentParser",
    "Embedder",
    "Healthcheck",
    "LexicalIndex",
    "Reranker",
    "Retriever",
    "VectorSearcher",
    "VectorWriter",
]


class Embedder(Protocol):
    """Turns text into vectors. Raises ``EmbeddingFailed`` / ``UpstreamRateLimited``."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per input, same order, each of length ``dimension``."""
        ...

    async def embed_query(self, text: str) -> list[float]: ...


class VectorWriter(Protocol):
    async def ensure_ready(self, *, dimension: int, embedding_model: str) -> None:
        """Create the index if absent. Raises ``IndexIncompatible`` if it exists with another model/size."""
        ...

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        embedding_model: str,
    ) -> None:
        """Idempotent by chunk id. Raises ``VectorStoreUnavailable`` / ``VectorStoreError``."""
        ...

    async def delete_document(self, document_id: str, *, keep_chunk_ids: Collection[str] = ()) -> int:
        """Delete the document's chunks except ``keep_chunk_ids``; return how many were deleted."""
        ...


class VectorSearcher(Protocol):
    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filter: ChunkFilter | None = None,
    ) -> list[ScoredChunk]:
        """Best first; ``score`` is similarity (higher is better)."""
        ...


class DocumentCatalog(Protocol):
    async def get_document(self, document_id: str) -> DocumentRecord | None: ...

    async def list_documents(self, *, offset: int, limit: int) -> list[DocumentRecord]:
        """Stable order (by id)."""
        ...


class ChunkScanner(Protocol):
    def scan(self, *, batch_size: int) -> AsyncIterator[list[Chunk]]:
        """Every stored chunk, in batches. Used to rebuild in-memory indexes at start-up."""
        ...


class LexicalIndex(Protocol):
    """Keyword (BM25) retrieval over chunks. Synchronous and in-memory by contract."""

    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def remove_document(self, document_id: str) -> None: ...

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filter: ChunkFilter | None = None,
    ) -> list[ScoredChunk]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int) -> list[ScoredChunk]:
        """Best first, at most ``top_k``, only elements of ``candidates``."""
        ...


class ChatModel(Protocol):
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> Completion:
        """Raises ``ModelFailed`` / ``UpstreamRateLimited``."""
        ...


@runtime_checkable
class DocumentParser(Protocol):
    """Parses one family of files. CPU-bound: callers run it in a worker thread."""

    @property
    def extensions(self) -> frozenset[str]:
        """Lower-case, with the dot, for example ``{".pdf"}``."""
        ...

    def parse(self, data: bytes, *, name: str) -> ParsedDocument:
        """Raises ``DocumentParseFailed`` (corrupt/unreadable) or ``DocumentEmpty`` (no text)."""
        ...


class Chunker(Protocol):
    @property
    def version(self) -> str:
        """Part of every chunk id: changing chunking behaviour must change this."""
        ...

    def chunk(self, doc: ParsedDocument, *, document_id: str, document_name: str) -> list[Chunk]: ...


class Retriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filter: ChunkFilter | None = None,
    ) -> list[ScoredChunk]: ...


class AnswerPipeline(Protocol):
    """Produces an answer from retrieval. Implementations: direct (plain calls), crewai (a crew)."""

    @property
    def name(self) -> str: ...

    async def answer(
        self,
        question: str,
        *,
        retriever: Retriever,
        top_k: int,
        filter: ChunkFilter | None = None,
    ) -> PipelineAnswer:
        """Cited chunk ids must come from what the pipeline retrieved. Raises ``ModelOutputInvalid``."""
        ...


class Healthcheck(Protocol):
    async def check(self) -> None:
        """Return normally if the dependency is reachable; raise a ``RagError`` otherwise."""
        ...
