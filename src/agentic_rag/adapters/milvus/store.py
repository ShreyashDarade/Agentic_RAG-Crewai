from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException, MilvusUnavailableException

from agentic_rag.adapters.milvus.filters import quote_ids, render
from agentic_rag.errors import (
    IndexIncompatible,
    LimitExceeded,
    VectorStoreError,
    VectorStoreUnavailable,
)
from agentic_rag.ports import Chunk, ChunkFilter, DocumentRecord, ScoredChunk

__all__ = ["MilvusSettings", "MilvusStore"]

T = TypeVar("T")

_FIELDS = ["id", "document_id", "chunk_index", "page", "text", "document_name", "embedding_model", "metadata"]
_MAX_DOCUMENTS_LISTED = 16_000
_UNAVAILABLE_CODES = {2}  # pymilvus: "Fail connecting to server"


@dataclass(frozen=True, slots=True)
class MilvusSettings:
    uri: str
    token: str | None = None
    collection: str = "documents"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    search_ef: int = 64
    consistency_level: str = "Strong"


class MilvusStore:
    """Chunks live in one collection; the document record is the metadata of chunk 0 (commit marker)."""

    def __init__(self, settings: MilvusSettings) -> None:
        self._s = settings
        self._client_obj: MilvusClient | None = None
        self._connect_lock = threading.Lock()
        # Milvus Lite (a local file URI) is single-process and not safe for concurrent calls.
        self._serial = threading.Lock() if not settings.uri.startswith(("http://", "https://")) else None
        self._exists = False

    # -- plumbing --------------------------------------------------------------------------------

    def _client(self) -> MilvusClient:
        with self._connect_lock:
            if self._client_obj is None:
                self._client_obj = MilvusClient(uri=self._s.uri, token=self._s.token or "")
            return self._client_obj

    async def _run(self, fn: Callable[[MilvusClient], T]) -> T:
        def call() -> T:
            try:
                client = self._client()
                if self._serial is None:
                    return fn(client)
                with self._serial:
                    return fn(client)
            except MilvusException as exc:
                if isinstance(exc, MilvusUnavailableException) or exc.code in _UNAVAILABLE_CODES:
                    raise VectorStoreUnavailable() from exc
                raise VectorStoreError() from exc

        return await asyncio.to_thread(call)

    def _collection_exists(self, client: MilvusClient) -> bool:
        if not self._exists:
            self._exists = bool(client.has_collection(self._s.collection))
        return self._exists

    # -- Healthcheck -----------------------------------------------------------------------------

    async def check(self) -> None:
        await self._run(lambda c: c.list_collections())

    # -- VectorWriter ----------------------------------------------------------------------------

    async def ensure_ready(self, *, dimension: int, embedding_model: str) -> None:
        def work(c: MilvusClient) -> None:
            if not self._collection_exists(c):
                self._create(c, dimension)
                self._exists = True
                return
            actual = _vector_dimension(c.describe_collection(self._s.collection))
            if actual != dimension:
                raise IndexIncompatible(details={"index_dimension": actual, "requested": dimension})
            rows = c.query(
                self._s.collection, filter="chunk_index >= 0", output_fields=["embedding_model"], limit=1
            )
            if rows and rows[0].get("embedding_model") != embedding_model:
                raise IndexIncompatible(details={"index_model": rows[0].get("embedding_model")})

        await self._run(work)

    def _create(self, c: MilvusClient, dimension: int) -> None:
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("document_id", DataType.VARCHAR, max_length=128)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("page", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("document_name", DataType.VARCHAR, max_length=255)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=128)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        index = c.prepare_index_params()
        index.add_index(
            "vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": self._s.hnsw_m, "efConstruction": self._s.hnsw_ef_construction},
        )
        c.create_collection(
            self._s.collection, schema=schema, index_params=index, consistency_level=self._s.consistency_level
        )

    async def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, embedding_model: str
    ) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunk and vector counts differ")
        rows = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.index,
                "page": -1 if chunk.page is None else chunk.page,
                "text": chunk.text,
                "document_name": chunk.document_name,
                "embedding_model": embedding_model,
                "metadata": dict(chunk.metadata),
                "vector": list(vector),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        def work(c: MilvusClient) -> None:
            c.upsert(self._s.collection, rows)

        await self._run(work)

    async def delete_document(self, document_id: str, *, keep_chunk_ids: Collection[str] = ()) -> int:
        expr = f"document_id == {quote_ids([document_id])[1:-1]}"
        if keep_chunk_ids:
            expr += f" and id not in {quote_ids(sorted(keep_chunk_ids))}"

        def work(c: MilvusClient) -> int:
            if not self._collection_exists(c):
                return 0
            deleted = c.delete(self._s.collection, filter=expr)
            return len(deleted) if isinstance(deleted, list) else int(deleted.get("delete_count", 0))

        return await self._run(work)

    # -- VectorSearcher --------------------------------------------------------------------------

    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filter: ChunkFilter | None = None,
    ) -> list[ScoredChunk]:
        expr = render(filter)

        def work(c: MilvusClient) -> list[ScoredChunk]:
            if not self._collection_exists(c):
                return []
            result = c.search(
                self._s.collection,
                data=[list(vector)],
                limit=top_k,
                filter=expr,
                output_fields=_FIELDS,
                search_params={"params": {"ef": max(self._s.search_ef, top_k * 2)}},
            )
            return [ScoredChunk(_chunk(hit["entity"]), float(hit["distance"])) for hit in result[0]]

        return await self._run(work)

    # -- DocumentCatalog -------------------------------------------------------------------------

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        expr = f"document_id == {quote_ids([document_id])[1:-1]} and chunk_index == 0"

        def work(c: MilvusClient) -> DocumentRecord | None:
            if not self._collection_exists(c):
                return None
            rows = c.query(self._s.collection, filter=expr, output_fields=_FIELDS, limit=1)
            return _record(rows[0]) if rows else None

        return await self._run(work)

    async def list_documents(self, *, offset: int, limit: int) -> list[DocumentRecord]:
        def work(c: MilvusClient) -> list[DocumentRecord]:
            if not self._collection_exists(c):
                return []
            rows = c.query(
                self._s.collection,
                filter="chunk_index == 0",
                output_fields=_FIELDS,
                limit=_MAX_DOCUMENTS_LISTED,
            )
            if len(rows) >= _MAX_DOCUMENTS_LISTED:
                raise LimitExceeded("too many documents to list in one collection")
            records = sorted((_record(r) for r in rows), key=lambda r: r.id)
            return records[offset : offset + limit]

        return await self._run(work)

    # -- ChunkScanner ----------------------------------------------------------------------------

    async def scan(self, *, batch_size: int) -> AsyncIterator[list[Chunk]]:
        iterator = await self._run(self._open_iterator(batch_size))
        if iterator is None:
            return
        try:
            while True:
                batch = await self._run(_next(iterator))
                if not batch:
                    return
                yield [_chunk(row) for row in batch]
        finally:
            await self._run(_close(iterator))

    def _open_iterator(self, batch_size: int) -> Callable[[MilvusClient], Any]:
        def work(c: MilvusClient) -> Any:
            if not self._collection_exists(c):
                return None
            return c.query_iterator(
                self._s.collection, batch_size=batch_size, filter="chunk_index >= 0", output_fields=_FIELDS
            )

        return work


def _next(iterator: Any) -> Callable[[MilvusClient], list[dict[str, Any]]]:
    def work(_client: MilvusClient) -> list[dict[str, Any]]:
        return list(iterator.next())

    return work


def _close(iterator: Any) -> Callable[[MilvusClient], None]:
    def work(_client: MilvusClient) -> None:
        iterator.close()

    return work


def _vector_dimension(description: dict[str, Any]) -> int:
    for field in description.get("fields", []):
        if field.get("name") == "vector":
            return int(field["params"]["dim"])
    raise IndexIncompatible("the existing collection has no vector field")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata")
    if isinstance(meta, str):
        meta = json.loads(meta)
    return dict(meta or {})


def _chunk(row: dict[str, Any]) -> Chunk:
    page = int(row.get("page", -1))
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        index=int(row["chunk_index"]),
        text=row["text"],
        document_name=row.get("document_name", ""),
        page=None if page < 0 else page,
        metadata=_metadata(row),
    )


def _record(row: dict[str, Any]) -> DocumentRecord:
    meta = _metadata(row)
    return DocumentRecord(
        id=row["document_id"],
        name=row.get("document_name", ""),
        content_sha256=str(meta.get("content_sha256", "")),
        content_type=str(meta.get("content_type", "")),
        chunk_count=int(meta.get("chunk_count", 0)),
        embedding_model=row.get("embedding_model", ""),
    )
