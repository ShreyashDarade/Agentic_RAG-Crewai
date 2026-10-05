"""Qdrant vector store (local path, in-memory, or a server). ``qdrant_client`` is imported only here.

Point ids are UUIDv5 of the chunk id (Qdrant ids must be UUIDs or integers); the chunk id lives in the payload.
Filters are typed condition objects, never expression strings.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from agentic_rag.blocking import STORE_BACKLOG, STORE_WORKERS, BlockingPool
from agentic_rag.errors import IndexIncompatible, LimitExceeded, VectorStoreError, VectorStoreUnavailable
from agentic_rag.ports import Chunk, ChunkFilter, DocumentRecord, ScoredChunk

__all__ = ["QdrantSettings", "QdrantStore"]

T = TypeVar("T")
_BATCH = 500
_MAX_DOCUMENTS_LISTED = 16_000
_NAMESPACE = uuid.UUID("6f1c5c2e-4a0b-4b7e-9d53-1d6f0a0f6a11")


@dataclass(frozen=True, slots=True)
class QdrantSettings:
    location: str  # ":memory:", a local directory path, or http(s)://host:port
    api_key: str | None = None
    collection: str = "documents"
    timeout_seconds: float = 20.0


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantStore:
    def __init__(self, settings: QdrantSettings) -> None:
        self._s = settings
        self._client_obj: QdrantClient | None = None
        self._lock = threading.RLock() if not settings.location.startswith(("http://", "https://")) else None
        self._pool = BlockingPool(
            "qdrant",
            workers=STORE_WORKERS,
            backlog=STORE_BACKLOG,
            saturated=lambda: VectorStoreUnavailable("the store is not answering"),
        )

    def _client(self) -> QdrantClient:
        if self._client_obj is None:
            loc = self._s.location
            if loc.startswith(("http://", "https://")):
                self._client_obj = QdrantClient(
                    url=loc,
                    api_key=self._s.api_key,
                    timeout=max(1, round(self._s.timeout_seconds)),
                    check_compatibility=False,
                )
            elif loc == ":memory:":
                self._client_obj = QdrantClient(location=":memory:")
            else:
                self._client_obj = QdrantClient(path=loc)
        return self._client_obj

    async def _run(self, fn: Callable[[QdrantClient], T]) -> T:
        def call() -> T:
            try:
                if self._lock is None:
                    return fn(self._client())
                with self._lock:
                    return fn(self._client())
            except ResponseHandlingException as exc:
                raise VectorStoreUnavailable() from exc
            except UnexpectedResponse as exc:
                raise VectorStoreError() from exc

        return await self._pool.run(call)

    def _exists(self, c: QdrantClient) -> bool:
        return bool(c.collection_exists(self._s.collection))

    async def aclose(self) -> None:
        client, self._client_obj = self._client_obj, None
        try:
            if client is not None:
                async with asyncio.timeout(5):
                    await self._pool.run(client.close)
        finally:
            self._pool.close()

    async def check(self) -> None:
        await self._run(lambda c: c.get_collections())

    async def ensure_ready(self, *, dimension: int, embedding_model: str) -> None:
        def work(c: QdrantClient) -> None:
            if not self._exists(c):
                c.create_collection(
                    self._s.collection,
                    vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                )
                return
            vectors = c.get_collection(self._s.collection).config.params.vectors
            size = vectors.size if isinstance(vectors, models.VectorParams) else None
            if size != dimension:
                raise IndexIncompatible(details={"index_dimension": size, "requested": dimension})
            points, _ = c.scroll(self._s.collection, limit=1, with_payload=["embedding_model"])
            if points and (points[0].payload or {}).get("embedding_model") != embedding_model:
                raise IndexIncompatible(details={"index_model": (points[0].payload or {}).get("embedding_model")})

        await self._run(work)

    async def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, embedding_model: str
    ) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunk and vector counts differ")

        def work(c: QdrantClient) -> None:
            for start in range(0, len(chunks), _BATCH):
                c.upsert(
                    self._s.collection,
                    points=[
                        models.PointStruct(id=_point_id(x.id), vector=list(v), payload=_payload(x, embedding_model))
                        for x, v in zip(chunks[start : start + _BATCH], vectors[start : start + _BATCH], strict=True)
                    ],
                )

        await self._run(work)

    async def delete_document(self, document_id: str, *, keep_chunk_ids: Collection[str] = ()) -> int:
        flt = models.Filter(
            must=[_match("document_id", [document_id])],
            must_not=[models.HasIdCondition(has_id=[_point_id(i) for i in keep_chunk_ids])] if keep_chunk_ids else None,
        )

        def work(c: QdrantClient) -> int:
            if not self._exists(c):
                return 0
            doomed = c.count(self._s.collection, count_filter=flt, exact=True).count
            if doomed:
                c.delete(self._s.collection, points_selector=models.FilterSelector(filter=flt))
            return int(doomed)

        return await self._run(work)

    async def search(
        self, vector: Sequence[float], *, top_k: int, filter: ChunkFilter | None = None
    ) -> list[ScoredChunk]:
        flt = _filter(filter)

        def work(c: QdrantClient) -> list[ScoredChunk]:
            if not self._exists(c):
                return []
            result = c.query_points(
                self._s.collection, query=list(vector), limit=top_k, query_filter=flt, with_payload=True
            )
            return [ScoredChunk(_to_chunk(p.payload or {}), float(p.score)) for p in result.points]

        return await self._run(work)

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        flt = models.Filter(must=[_match("document_id", [document_id]), _match("chunk_index", 0)])

        def work(c: QdrantClient) -> DocumentRecord | None:
            if not self._exists(c):
                return None
            points, _ = c.scroll(self._s.collection, scroll_filter=flt, limit=1, with_payload=True)
            return _to_record(points[0].payload or {}) if points else None

        return await self._run(work)

    async def list_documents(self, *, offset: int, limit: int) -> list[DocumentRecord]:
        flt = models.Filter(must=[_match("chunk_index", 0)])

        def work(c: QdrantClient) -> list[DocumentRecord]:
            if not self._exists(c):
                return []
            points, _ = c.scroll(self._s.collection, scroll_filter=flt, limit=_MAX_DOCUMENTS_LISTED, with_payload=True)
            if len(points) >= _MAX_DOCUMENTS_LISTED:
                raise LimitExceeded("too many documents to list in one collection")
            records = sorted((_to_record(p.payload or {}) for p in points), key=lambda r: r.id)
            return records[offset : offset + limit]

        return await self._run(work)

    async def scan(self, *, batch_size: int) -> AsyncIterator[list[Chunk]]:
        cursor: Any = None
        while True:

            def work(c: QdrantClient, after: Any = cursor) -> tuple[list[Chunk], Any]:
                if not self._exists(c):
                    return [], None
                points, nxt = c.scroll(self._s.collection, limit=batch_size, offset=after, with_payload=True)
                return [_to_chunk(p.payload or {}) for p in points], nxt

            batch, cursor = await self._run(work)
            if not batch:
                return
            yield batch
            if cursor is None:
                return


def _match(key: str, value: Any) -> models.FieldCondition:
    if isinstance(value, list):
        return models.FieldCondition(key=key, match=models.MatchAny(any=value))
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _filter(flt: ChunkFilter | None) -> models.Filter | None:
    if flt is None or not flt.document_ids:
        return None
    return models.Filter(must=[_match("document_id", list(flt.document_ids))])


def _payload(chunk: Chunk, model: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.index,
        "page": -1 if chunk.page is None else chunk.page,
        "text": chunk.text,
        "document_name": chunk.document_name,
        "embedding_model": model,
        "metadata": dict(chunk.metadata),
    }


def _to_chunk(p: dict[str, Any]) -> Chunk:
    page = int(p.get("page", -1))
    return Chunk(
        id=str(p["chunk_id"]),
        document_id=str(p["document_id"]),
        index=int(p["chunk_index"]),
        text=str(p["text"]),
        document_name=str(p.get("document_name", "")),
        page=None if page < 0 else page,
        metadata=dict(p.get("metadata", {})),
    )


def _to_record(p: dict[str, Any]) -> DocumentRecord:
    meta = p.get("metadata", {})
    return DocumentRecord(
        id=str(p["document_id"]),
        name=str(p.get("document_name", "")),
        content_sha256=str(meta.get("content_sha256", "")),
        content_type=str(meta.get("content_type", "")),
        chunk_count=int(meta.get("chunk_count", 0)),
        index_version=str(meta.get("index_version", "")),
        embedding_model=str(p.get("embedding_model", "")),
    )
