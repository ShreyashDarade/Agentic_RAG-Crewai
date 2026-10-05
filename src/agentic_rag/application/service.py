"""The application service: every use case exactly once (ADR-0001).

Transports (HTTP routes, the in-process SDK backend, the CLI) call these methods and nothing else.
The service depends only on ports, never on a framework or an adapter.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import dataclasses
import hashlib
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import PurePosixPath

from agentic_rag.application.limits import Limits, RetrievalConfig
from agentic_rag.contracts import (
    DeleteResult,
    DocumentInfo,
    DocumentList,
    IngestResult,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    Source,
)
from agentic_rag.errors import (
    DeadlineExceeded,
    DocumentEmpty,
    DocumentNotFound,
    DocumentParseFailed,
    EmbeddingFailed,
    LimitExceeded,
    ModelOutputInvalid,
    PayloadTooLarge,
    RagError,
    UnsupportedFileType,
    ValidationFailed,
)
from agentic_rag.ports import (
    AnswerPipeline,
    Chunk,
    Chunker,
    ChunkFilter,
    DocumentCatalog,
    DocumentParser,
    DocumentRecord,
    Embedder,
    Healthcheck,
    LexicalIndex,
    ParsedDocument,
    Retriever,
    ScoredChunk,
    VectorWriter,
)

__all__ = ["Service", "sanitize_name"]

_SNIPPET_CHARS = 300
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_name(name: str) -> str:
    """A display name: no directory parts, no control characters, at most 255 characters."""
    base = PurePosixPath(name.replace("\\", "/")).name
    base = _CONTROL.sub("", base).strip()
    if not base or base in {".", ".."}:
        raise ValidationFailed("the file name is empty")
    return base[:255]


class Service:
    def __init__(
        self,
        *,
        embedder: Embedder,
        writer: VectorWriter,
        catalog: DocumentCatalog,
        retriever: Retriever,
        lexical: LexicalIndex | None,
        parsers: Sequence[DocumentParser],
        chunker: Chunker,
        pipeline: AnswerPipeline,
        limits: Limits,
        retrieval: RetrievalConfig,
        health_checks: Mapping[str, Healthcheck],
    ) -> None:
        self._embedder = embedder
        self._writer = writer
        self._catalog = catalog
        self._retriever = retriever
        self._lexical = lexical
        self._parsers: dict[str, DocumentParser] = {}
        for parser in parsers:
            for ext in parser.extensions:
                self._parsers[ext] = parser
        self._chunker = chunker
        self._pipeline = pipeline
        self._limits = limits
        self._retrieval = retrieval
        self._health = dict(health_checks)
        self._ingest_slots = asyncio.Semaphore(limits.max_concurrent_ingests)
        self._doc_locks: dict[str, asyncio.Lock] = {}
        self._prepared = False

    # -- queries ---------------------------------------------------------------------------------

    async def query(self, request: QueryRequest) -> QueryResponse:
        question = self._check_text(request.question)
        top_k = self._top_k(request.top_k)
        async with self._deadline():
            result = await self._pipeline.answer(
                question,
                retriever=self._retriever,
                top_k=top_k,
                filter=self._filter(request.document_ids),
            )
        retrieved = {hit.chunk.id for hit in result.retrieved}
        unknown = [cid for cid in result.cited_chunk_ids if cid not in retrieved]
        if unknown:
            raise ModelOutputInvalid(
                "the answer cited chunks that were not retrieved",
                details={"unknown_citation_count": len(unknown)},
            )
        citations = list(dict.fromkeys(result.cited_chunk_ids))
        return QueryResponse(
            answer=result.text,
            grounded=bool(citations),
            citations=citations,
            sources=[self._source(hit) for hit in result.retrieved],
            pipeline=self._pipeline.name,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        query = self._check_text(request.query)
        async with self._deadline():
            hits = await self._retriever.retrieve(
                query,
                top_k=self._top_k(request.top_k),
                filter=self._filter(request.document_ids),
            )
        return SearchResponse(
            hits=[
                SearchHit(
                    chunk_id=h.chunk.id,
                    document_id=h.chunk.document_id,
                    document_name=h.chunk.document_name,
                    text=h.chunk.text,
                    score=h.score,
                    page=h.chunk.page,
                )
                for h in hits
            ]
        )

    # -- documents -------------------------------------------------------------------------------

    async def ingest_document(self, name: str, data: bytes) -> IngestResult:
        safe_name = sanitize_name(name)
        extension = PurePosixPath(safe_name).suffix.lower()
        parser = self._parsers.get(extension)
        if parser is None:
            raise UnsupportedFileType(details={"extension": extension, "supported": sorted(self._parsers)})
        if len(data) > self._limits.max_upload_bytes:
            raise PayloadTooLarge(details={"max_bytes": self._limits.max_upload_bytes})
        if not data:
            raise DocumentEmpty("the file is empty")
        sha = hashlib.sha256(data).hexdigest()
        document_id = f"doc_{sha[:32]}"
        async with self._deadline(), self._ingest_slots, self._doc_lock(document_id):
            existing = await self._catalog.get_document(document_id)
            if existing is not None and existing.embedding_model == self._embedder.model_id:
                return IngestResult(document=_info(existing), created=False, chunks_indexed=existing.chunk_count)
            parsed = await self._parse(parser, data, safe_name)
            chunks = await asyncio.to_thread(
                self._chunker.chunk, parsed, document_id=document_id, document_name=safe_name
            )
            if not chunks:
                raise DocumentEmpty()
            if len(chunks) > self._limits.max_chunks_per_document:
                raise LimitExceeded(details={"max_chunks_per_document": self._limits.max_chunks_per_document})
            chunks = _annotate(chunks, sha=sha, extension=extension)
            vectors = await self._embed(chunks)
            await self._writer.ensure_ready(dimension=self._embedder.dimension, embedding_model=self._embedder.model_id)
            # Two phases: every chunk except chunk 0, then chunk 0. Chunk 0 carries the document record, so it is the
            # commit marker: a crash before it leaves no catalog entry, and a retry (same content, same ids) finishes.
            body = [i for i, chunk in enumerate(chunks) if chunk.index != 0]
            head = [i for i, chunk in enumerate(chunks) if chunk.index == 0]
            for phase in (body, head):
                if phase:
                    await self._writer.upsert(
                        [chunks[i] for i in phase],
                        [vectors[i] for i in phase],
                        embedding_model=self._embedder.model_id,
                    )
            await self._writer.delete_document(document_id, keep_chunk_ids={chunk.id for chunk in chunks})
            if self._lexical is not None:
                await asyncio.to_thread(self._reindex_lexical, document_id, chunks)
            record = DocumentRecord(
                id=document_id,
                name=safe_name,
                content_sha256=sha,
                content_type=extension,
                chunk_count=len(chunks),
                embedding_model=self._embedder.model_id,
            )
            return IngestResult(document=_info(record), created=True, chunks_indexed=len(chunks))

    async def get_document(self, document_id: str) -> DocumentInfo:
        record = await self._catalog.get_document(document_id)
        if record is None:
            raise DocumentNotFound()
        return _info(record)

    async def list_documents(self, *, limit: int = 50, page_token: str | None = None) -> DocumentList:
        if not 1 <= limit <= self._limits.max_page_size:
            raise ValidationFailed(details={"max_page_size": self._limits.max_page_size})
        offset = _decode_token(page_token)
        records = await self._catalog.list_documents(offset=offset, limit=limit + 1)
        page = records[:limit]
        more = len(records) > limit
        return DocumentList(
            items=[_info(r) for r in page],
            next_page_token=_encode_token(offset + limit) if more else None,
        )

    async def delete_document(self, document_id: str) -> DeleteResult:
        async with self._doc_lock(document_id):
            if await self._catalog.get_document(document_id) is None:
                raise DocumentNotFound()
            deleted = await self._writer.delete_document(document_id)
            if self._lexical is not None:
                await asyncio.to_thread(self._lexical.remove_document, document_id)
        return DeleteResult(document_id=document_id, chunks_deleted=deleted)

    # -- readiness -------------------------------------------------------------------------------

    async def ready(self) -> ReadyResponse:
        async def run(check: Healthcheck) -> str:
            try:
                async with asyncio.timeout(self._limits.health_check_timeout_seconds):
                    await check.check()
            except TimeoutError:
                return "timeout"
            except RagError:
                return "unavailable"
            return "ok"

        names = list(self._health)
        states = await asyncio.gather(*(run(self._health[n]) for n in names))
        checks = dict(zip(names, states, strict=True))
        return ReadyResponse(ready=all(s == "ok" for s in states), checks=checks)

    # -- helpers ---------------------------------------------------------------------------------

    @asynccontextmanager
    async def _deadline(self) -> AsyncIterator[None]:
        timeout = asyncio.timeout(self._limits.request_deadline_seconds)
        try:
            async with timeout:
                yield
        except TimeoutError as exc:
            if timeout.expired():
                raise DeadlineExceeded() from exc
            raise

    def _doc_lock(self, document_id: str) -> asyncio.Lock:
        # One lock per document id serialises concurrent ingests/deletes of the same content in this process.
        return self._doc_locks.setdefault(document_id, asyncio.Lock())

    def _check_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValidationFailed("the question is empty")
        if len(text) > self._limits.max_question_chars:
            raise LimitExceeded(details={"max_question_chars": self._limits.max_question_chars})
        return text

    def _top_k(self, requested: int | None) -> int:
        top_k = requested if requested is not None else self._retrieval.default_top_k
        if top_k > self._limits.max_top_k:
            raise LimitExceeded(details={"max_top_k": self._limits.max_top_k})
        return top_k

    @staticmethod
    def _filter(document_ids: Sequence[str] | None) -> ChunkFilter | None:
        return ChunkFilter(document_ids=tuple(document_ids)) if document_ids else None

    @staticmethod
    def _source(hit: ScoredChunk) -> Source:
        return Source(
            chunk_id=hit.chunk.id,
            document_id=hit.chunk.document_id,
            document_name=hit.chunk.document_name,
            snippet=hit.chunk.text[:_SNIPPET_CHARS],
            score=hit.score,
            page=hit.chunk.page,
        )

    async def _parse(self, parser: DocumentParser, data: bytes, name: str) -> ParsedDocument:
        try:
            return await asyncio.to_thread(parser.parse, data, name=name)
        except RagError:
            raise
        except Exception as exc:
            raise DocumentParseFailed() from exc

    async def _embed(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        batch = self._limits.embed_batch_size
        slots = asyncio.Semaphore(self._limits.max_concurrent_embed_batches)
        texts = [chunk.text for chunk in chunks]

        async def run(start: int) -> list[list[float]]:
            async with slots:
                return await self._embedder.embed_documents(texts[start : start + batch])

        parts = await asyncio.gather(*(run(i) for i in range(0, len(texts), batch)))
        vectors = [vector for part in parts for vector in part]
        if len(vectors) != len(chunks) or any(len(v) != self._embedder.dimension for v in vectors):
            raise EmbeddingFailed("the embedder returned the wrong number or size of vectors")
        return vectors

    def _reindex_lexical(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        assert self._lexical is not None
        self._lexical.remove_document(document_id)
        self._lexical.add(chunks)


def _annotate(chunks: Sequence[Chunk], *, sha: str, extension: str) -> list[Chunk]:
    extra: dict[str, str | int | float | bool] = {
        "content_sha256": sha,
        "content_type": extension,
        "chunk_count": len(chunks),
    }
    return [dataclasses.replace(c, metadata={**c.metadata, **extra}) for c in chunks]


def _info(record: DocumentRecord) -> DocumentInfo:
    return DocumentInfo(
        id=record.id,
        name=record.name,
        content_sha256=record.content_sha256,
        content_type=record.content_type,
        chunk_count=record.chunk_count,
        embedding_model=record.embedding_model,
    )


def _encode_token(offset: int) -> str:
    return base64.urlsafe_b64encode(f"o:{offset}".encode()).decode().rstrip("=")


def _decode_token(token: str | None) -> int:
    if token is None:
        return 0
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
        prefix, _, number = raw.partition(":")
        offset = int(number)
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise ValidationFailed("the page token is not valid") from None
    if prefix != "o" or offset < 0:
        raise ValidationFailed("the page token is not valid")
    return offset
