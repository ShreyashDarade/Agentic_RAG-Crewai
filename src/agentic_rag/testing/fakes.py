"""In-memory fakes for the ports: deterministic, dependency-free, usable by plug-in authors.

They are real implementations of the contracts (and pass the same conformance suites), not mocks.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator, Collection, Sequence

from agentic_rag.errors import EmbeddingFailed, IndexIncompatible, ModelFailed, VectorStoreUnavailable
from agentic_rag.ports import (
    ChatMessage,
    Chunk,
    ChunkFilter,
    Completion,
    DocumentRecord,
    ScoredChunk,
)

__all__ = ["FakeChatModel", "FakeEmbedder", "FakeReranker", "FakeRetriever", "FakeVectorStore"]


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class FakeEmbedder:
    """Hash-based bag-of-words vectors: texts sharing words are closer. Deterministic."""

    def __init__(self, dimension: int = 64, *, fail: bool = False) -> None:
        self._dimension = dimension
        self._fail = fail
        self.calls = 0

    @property
    def model_id(self) -> str:
        return f"fake-embed-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode()).digest()
            vec[int.from_bytes(digest[:4], "big") % self._dimension] += 1.0
        return _unit(vec)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self._fail:
            raise EmbeddingFailed()
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        if self._fail:
            raise EmbeddingFailed()
        return self._vector(text)


class FakeVectorStore:
    """Brute-force cosine store implementing writer, searcher, catalog, scanner and health check."""

    def __init__(self, *, unavailable: bool = False) -> None:
        self._rows: dict[str, tuple[Chunk, list[float], str]] = {}
        self._dimension: int | None = None
        self._unavailable = unavailable

    def _guard(self) -> None:
        if self._unavailable:
            raise VectorStoreUnavailable()

    async def check(self) -> None:
        self._guard()

    async def ensure_ready(self, *, dimension: int, embedding_model: str) -> None:
        self._guard()
        if self._dimension is not None and self._dimension != dimension:
            raise IndexIncompatible()
        models = {m for _, _, m in self._rows.values()}
        if models and models != {embedding_model}:
            raise IndexIncompatible()
        self._dimension = dimension

    async def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, embedding_model: str
    ) -> None:
        self._guard()
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._rows[chunk.id] = (chunk, list(vector), embedding_model)

    async def delete_document(self, document_id: str, *, keep_chunk_ids: Collection[str] = ()) -> int:
        self._guard()
        doomed = [
            cid
            for cid, (chunk, _, _) in self._rows.items()
            if chunk.document_id == document_id and cid not in keep_chunk_ids
        ]
        for cid in doomed:
            del self._rows[cid]
        return len(doomed)

    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filter: ChunkFilter | None = None,
    ) -> list[ScoredChunk]:
        self._guard()
        allowed = filter.document_ids if filter and filter.document_ids else None
        scored = [
            ScoredChunk(chunk, sum(a * b for a, b in zip(vector, vec, strict=True)))
            for chunk, vec, _ in self._rows.values()
            if allowed is None or chunk.document_id in allowed
        ]
        scored.sort(key=lambda s: (-s.score, s.chunk.id))
        return scored[:top_k]

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        self._guard()
        for chunk, _, model in self._rows.values():
            if chunk.document_id == document_id and chunk.index == 0:
                return _record(chunk, model)
        return None

    async def list_documents(self, *, offset: int, limit: int) -> list[DocumentRecord]:
        self._guard()
        heads = sorted(((c, m) for c, _, m in self._rows.values() if c.index == 0), key=lambda cm: cm[0].document_id)
        return [_record(c, m) for c, m in heads[offset : offset + limit]]

    async def scan(self, *, batch_size: int) -> AsyncIterator[list[Chunk]]:
        self._guard()
        chunks = sorted((c for c, _, _ in self._rows.values()), key=lambda c: c.id)
        for i in range(0, len(chunks), batch_size):
            yield chunks[i : i + batch_size]


def _record(chunk: Chunk, model: str) -> DocumentRecord:
    meta = chunk.metadata
    return DocumentRecord(
        id=chunk.document_id,
        name=chunk.document_name,
        content_sha256=str(meta.get("content_sha256", "")),
        content_type=str(meta.get("content_type", "")),
        chunk_count=int(meta.get("chunk_count", 0)),
        embedding_model=model,
    )


class FakeChatModel:
    """Returns scripted replies in order (the last one repeats); records every call."""

    def __init__(self, replies: Sequence[str] = ("ok",), *, fail: bool = False) -> None:
        self._replies = list(replies)
        self._fail = fail
        self.calls: list[list[ChatMessage]] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> Completion:
        self.calls.append(list(messages))
        if self._fail:
            raise ModelFailed()
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        return Completion(text=self._replies[index])


class FakeRetriever:
    """Returns fixed hits (ignoring the query) and records calls."""

    def __init__(self, hits: Sequence[ScoredChunk] = ()) -> None:
        self._hits = list(hits)
        self.calls: list[tuple[str, int, ChunkFilter | None]] = []

    async def retrieve(self, query: str, *, top_k: int, filter: ChunkFilter | None = None) -> list[ScoredChunk]:
        self.calls.append((query, top_k, filter))
        return self._hits[:top_k]


class FakeReranker:
    """Reorders candidates by word overlap with the query (stable, dependency-free)."""

    async def rerank(self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int) -> list[ScoredChunk]:
        words = set(query.lower().split())

        def overlap(hit: ScoredChunk) -> int:
            return len(words & set(hit.chunk.text.lower().split()))

        ranked = sorted(candidates, key=lambda h: (-overlap(h), -h.score, h.chunk.id))
        return [ScoredChunk(h.chunk, float(overlap(h))) for h in ranked[:top_k]]
