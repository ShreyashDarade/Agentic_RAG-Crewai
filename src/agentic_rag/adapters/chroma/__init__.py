"""Chroma vector store (local persistent directory or a Chroma server). ``chromadb`` is imported only here.

Same port and conformance suite as the Milvus adapter. Filters are typed ``where`` dictionaries, never strings.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import urlsplit

import chromadb
from chromadb.errors import ChromaError, NotFoundError

from agentic_rag.errors import IndexIncompatible, LimitExceeded, VectorStoreError, VectorStoreUnavailable
from agentic_rag.ports import Chunk, ChunkFilter, DocumentRecord, ScoredChunk

__all__ = ["ChromaSettings", "ChromaStore"]

T = TypeVar("T")
_BATCH = 500
_MAX_DOCUMENTS_LISTED = 16_000
_INCLUDE = ["documents", "metadatas"]


@dataclass(frozen=True, slots=True)
class ChromaSettings:
    path: str | None = None  # a local persistent directory ...
    url: str | None = None  # ... or http://host:port of a Chroma server
    collection: str = "documents"


class ChromaStore:
    def __init__(self, settings: ChromaSettings) -> None:
        if (settings.path is None) == (settings.url is None):
            raise ValueError("set exactly one of path and url")
        self._s = settings
        self._client_obj: Any = None
        self._lock = threading.RLock()  # the local client is one process; serialise to be safe

    def _client(self) -> Any:
        if self._client_obj is None:
            if self._s.url:
                parts = urlsplit(self._s.url)
                self._client_obj = chromadb.HttpClient(
                    host=parts.hostname or "localhost", port=parts.port or 8000, ssl=parts.scheme == "https"
                )
            else:
                self._client_obj = chromadb.PersistentClient(path=str(self._s.path))
        return self._client_obj

    async def _run(self, fn: Callable[[Any], T]) -> T:
        def call() -> T:
            try:
                with self._lock:
                    return fn(self._client())
            except (ConnectionError, TimeoutError) as exc:
                raise VectorStoreUnavailable() from exc
            except ValueError as exc:  # chromadb raises ValueError("Could not connect to a Chroma server...")
                if "connect" in str(exc).lower():
                    raise VectorStoreUnavailable() from exc
                raise VectorStoreError() from exc
            except ChromaError as exc:
                raise VectorStoreError() from exc

        return await asyncio.to_thread(call)

    def _collection(self, client: Any) -> Any | None:
        try:
            return client.get_collection(self._s.collection)
        except (NotFoundError, ValueError, ChromaError) as exc:
            if isinstance(exc, NotFoundError) or "does not exist" in str(exc).lower():
                return None
            raise

    async def aclose(self) -> None:
        self._client_obj = None

    async def check(self) -> None:
        await self._run(lambda c: c.heartbeat())

    async def ensure_ready(self, *, dimension: int, embedding_model: str) -> None:
        def work(c: Any) -> None:
            col = c.get_or_create_collection(
                self._s.collection,
                metadata={"hnsw:space": "cosine", "embedding_model": embedding_model, "dimension": dimension},
            )
            meta = col.metadata or {}
            if meta.get("dimension") != dimension or meta.get("embedding_model") != embedding_model:
                raise IndexIncompatible(details={"index_dimension": meta.get("dimension")})

        await self._run(work)

    async def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, embedding_model: str
    ) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunk and vector counts differ")

        def work(c: Any) -> None:
            col = c.get_collection(self._s.collection)
            for start in range(0, len(chunks), _BATCH):
                part = chunks[start : start + _BATCH]
                col.upsert(
                    ids=[x.id for x in part],
                    embeddings=[list(v) for v in vectors[start : start + _BATCH]],
                    documents=[x.text for x in part],
                    metadatas=[_to_metadata(x, embedding_model) for x in part],
                )

        await self._run(work)

    async def delete_document(self, document_id: str, *, keep_chunk_ids: Collection[str] = ()) -> int:
        def work(c: Any) -> int:
            col = self._collection(c)
            if col is None:
                return 0
            ids = col.get(where={"document_id": document_id}, include=[])["ids"]
            doomed = [i for i in ids if i not in keep_chunk_ids]
            for start in range(0, len(doomed), _BATCH):
                col.delete(ids=doomed[start : start + _BATCH])
            return len(doomed)

        return await self._run(work)

    async def search(
        self, vector: Sequence[float], *, top_k: int, filter: ChunkFilter | None = None
    ) -> list[ScoredChunk]:
        where = _where(filter)

        def work(c: Any) -> list[ScoredChunk]:
            col = self._collection(c)
            if col is None or col.count() == 0:
                return []
            res = col.query(
                query_embeddings=[list(vector)],
                n_results=min(top_k, col.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            return [
                ScoredChunk(_to_chunk(i, doc, meta), 1.0 - float(dist))  # cosine distance -> similarity
                for i, doc, meta, dist in zip(
                    res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
                )
            ]

        return await self._run(work)

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        def work(c: Any) -> DocumentRecord | None:
            col = self._collection(c)
            if col is None:
                return None
            res = col.get(where={"$and": [{"document_id": document_id}, {"chunk_index": 0}]}, include=_INCLUDE)
            return _to_record(res["metadatas"][0]) if res["ids"] else None

        return await self._run(work)

    async def list_documents(self, *, offset: int, limit: int) -> list[DocumentRecord]:
        def work(c: Any) -> list[DocumentRecord]:
            col = self._collection(c)
            if col is None:
                return []
            res = col.get(where={"chunk_index": 0}, include=["metadatas"], limit=_MAX_DOCUMENTS_LISTED)
            if len(res["ids"]) >= _MAX_DOCUMENTS_LISTED:
                raise LimitExceeded("too many documents to list in one collection")
            records = sorted((_to_record(m) for m in res["metadatas"]), key=lambda r: r.id)
            return records[offset : offset + limit]

        return await self._run(work)

    async def scan(self, *, batch_size: int) -> AsyncIterator[list[Chunk]]:
        offset = 0
        while True:

            def work(c: Any, start: int = offset) -> list[Chunk]:
                col = self._collection(c)
                if col is None:
                    return []
                res = col.get(limit=batch_size, offset=start, include=_INCLUDE)
                return [
                    _to_chunk(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
                ]

            batch = await self._run(work)
            if not batch:
                return
            yield batch
            offset += len(batch)


def _to_metadata(chunk: Chunk, model: str) -> dict[str, str | int | float | bool]:
    meta: dict[str, str | int | float | bool] = {
        "document_id": chunk.document_id,
        "chunk_index": chunk.index,
        "page": -1 if chunk.page is None else chunk.page,
        "document_name": chunk.document_name,
        "embedding_model": model,
    }
    meta.update({f"m_{k}": v for k, v in chunk.metadata.items()})
    return meta


def _to_chunk(chunk_id: str, text: str, meta: dict[str, Any]) -> Chunk:
    page = int(meta.get("page", -1))
    return Chunk(
        id=chunk_id,
        document_id=str(meta["document_id"]),
        index=int(meta["chunk_index"]),
        text=text,
        document_name=str(meta.get("document_name", "")),
        page=None if page < 0 else page,
        metadata={k[2:]: v for k, v in meta.items() if k.startswith("m_")},
    )


def _to_record(meta: dict[str, Any]) -> DocumentRecord:
    return DocumentRecord(
        id=str(meta["document_id"]),
        name=str(meta.get("document_name", "")),
        content_sha256=str(meta.get("m_content_sha256", "")),
        content_type=str(meta.get("m_content_type", "")),
        chunk_count=int(meta.get("m_chunk_count", 0)),
        embedding_model=str(meta.get("embedding_model", "")),
    )


def _where(flt: ChunkFilter | None) -> dict[str, Any] | None:
    if flt is None or not flt.document_ids:
        return None
    return {"document_id": {"$in": list(flt.document_ids)}}
