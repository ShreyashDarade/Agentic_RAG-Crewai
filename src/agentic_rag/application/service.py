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
import logging
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import PurePosixPath

from agentic_rag.application.limits import Limits, RetrievalConfig
from agentic_rag.blocking import BlockingPool
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
    IndexIncompatible,
    LimitExceeded,
    ModelOutputInvalid,
    Overloaded,
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
    Retriever,
    ScoredChunk,
    VectorWriter,
)

__all__ = ["Service", "sanitize_name"]

logger = logging.getLogger("agentic_rag.service")

_SNIPPET_CHARS = 300
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_DOCUMENT_ID = re.compile(r"^doc_[0-9a-f]{32}$")  # the only shape this service ever issues
_CLEANUP_SECONDS = 5.0


def sanitize_name(name: str) -> str:
    """A display name: no directory parts, no control characters, at most 255 characters."""
    base = PurePosixPath(name.replace("\\", "/")).name
    base = _CONTROL.sub("", base).strip()
    if not base or base in {".", ".."}:
        raise ValidationFailed("the file name is empty")
    return base[:255]


@dataclasses.dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    users: int = 0


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
        self._doc_locks: dict[str, _LockEntry] = {}
        self._index_checked = False
        # Parsing and chunking are CPU-bound and cannot be interrupted. A cancelled request must not leave a pile of
        # them running, so a slot is held until the thread returns (see ``agentic_rag.blocking``).
        self._cpu = BlockingPool(
            "ingest-cpu",
            workers=limits.max_concurrent_ingests,
            backlog=limits.max_concurrent_ingests,
            saturated=Overloaded,
        )
        self._lexical_pool = BlockingPool("lexical", workers=2, backlog=256, saturated=Overloaded)

    def close(self) -> None:
        self._cpu.close()
        self._lexical_pool.close()

    # -- queries ---------------------------------------------------------------------------------

    async def query(self, request: QueryRequest) -> QueryResponse:
        question = self._check_text(request.question)
        top_k = self._top_k(request.top_k)
        async with self._deadline():
            await self._ensure_index()
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
            await self._ensure_index()
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
        # The id covers what the parser sees: the same bytes as .txt and as .html are different documents.
        document_id = "doc_" + hashlib.sha256(extension.encode() + b"\0" + data).hexdigest()[:32]
        index_version = f"{self._chunker.version}+{parser.version}"
        model = self._embedder.model_id
        async with self._deadline(), self._ingest_slots, self._doc_lock(document_id):
            existing = await self._catalog.get_document(document_id)
            if existing is not None and existing.embedding_model == model and existing.index_version == index_version:
                return IngestResult(document=_info(existing), created=False, chunks_indexed=existing.chunk_count)
            chunks = await self._cpu.run(self._parse_and_chunk, parser, data, safe_name, document_id)
            chunks = _annotate(chunks, sha=sha, extension=extension, index_version=index_version)
            vectors = await self._embed(chunks)
            await self._writer.ensure_ready(dimension=self._embedder.dimension, embedding_model=model)
            # write -> sweep -> commit (ADR-0007). Chunk 0 carries the document record, so it is the commit marker and
            # is written last: until it lands the document is not in the catalog and a retry (same ids) does the
            # whole job again rather than being told it already exists.
            body = [i for i, chunk in enumerate(chunks) if chunk.index != 0]
            head = [i for i, chunk in enumerate(chunks) if chunk.index == 0]
            try:
                if body:
                    await self._writer.upsert(
                        [chunks[i] for i in body], [vectors[i] for i in body], embedding_model=model
                    )
                await self._writer.delete_document(document_id, keep_chunk_ids={chunk.id for chunk in chunks})
                if self._lexical is not None:
                    await self._lexical_pool.run(self._reindex_lexical, document_id, chunks)
                await self._writer.upsert([chunks[i] for i in head], [vectors[i] for i in head], embedding_model=model)
            except BaseException:
                if existing is None:
                    await self._discard_uncommitted(document_id)
                raise
            record = DocumentRecord(
                id=document_id,
                name=safe_name,
                content_sha256=sha,
                content_type=extension,
                chunk_count=len(chunks),
                embedding_model=model,
                index_version=index_version,
            )
            return IngestResult(document=_info(record), created=True, chunks_indexed=len(chunks))

    async def get_document(self, document_id: str) -> DocumentInfo:
        _require_known_shape(document_id)
        async with self._deadline():
            record = await self._catalog.get_document(document_id)
        if record is None:
            raise DocumentNotFound()
        return _info(record)

    async def list_documents(self, *, limit: int = 50, page_token: str | None = None) -> DocumentList:
        if not 1 <= limit <= self._limits.max_page_size:
            raise ValidationFailed(details={"max_page_size": self._limits.max_page_size})
        offset = _decode_token(page_token)
        async with self._deadline():
            records = await self._catalog.list_documents(offset=offset, limit=limit + 1)
        page = records[:limit]
        more = len(records) > limit
        return DocumentList(
            items=[_info(r) for r in page],
            next_page_token=_encode_token(offset + limit) if more else None,
        )

    async def delete_document(self, document_id: str) -> DeleteResult:
        _require_known_shape(document_id)
        async with self._deadline(), self._doc_lock(document_id):
            record = await self._catalog.get_document(document_id)
            # Lexical first: if the store call then fails, the document is still fully findable and the delete can
            # simply be retried; the other order could leave deleted text searchable until restart.
            if self._lexical is not None:
                await self._lexical_pool.run(self._lexical.remove_document, document_id)
            deleted = await self._writer.delete_document(document_id)
            if record is None and deleted == 0:
                raise DocumentNotFound()  # chunks without a commit marker (an interrupted ingest) are still deletable
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

        async def index() -> str:
            try:
                async with asyncio.timeout(self._limits.health_check_timeout_seconds):
                    await self._ensure_index(force=True)
            except TimeoutError:
                return "timeout"
            except IndexIncompatible:
                return "incompatible"
            except RagError:
                return "unavailable"
            return "ok"

        names = [*self._health, "index"]
        states = await asyncio.gather(*(run(self._health[n]) if n in self._health else index() for n in names))
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

    @asynccontextmanager
    async def _doc_lock(self, document_id: str) -> AsyncIterator[None]:
        # One lock per document id serialises concurrent ingests/deletes of the same content in this process. The
        # entry is removed with its last user, so the table never outgrows the work in flight.
        entry = self._doc_locks.get(document_id)
        if entry is None:
            entry = self._doc_locks[document_id] = _LockEntry()
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                del self._doc_locks[document_id]

    async def _ensure_index(self, *, force: bool = False) -> None:
        """The index must have been built by this embedder. Cached after the first success on the read path;
        readiness re-checks every time so a mismatch introduced from outside is reported, not served."""
        if self._index_checked and not force:
            return
        self._index_checked = False
        await self._writer.ensure_ready(dimension=self._embedder.dimension, embedding_model=self._embedder.model_id)
        self._index_checked = True

    async def _discard_uncommitted(self, document_id: str) -> None:
        """Best effort, bounded: remove what a failed ingest wrote so it is not searchable without a catalog entry."""
        try:
            async with asyncio.timeout(_CLEANUP_SECONDS):
                if self._lexical is not None:
                    await self._lexical_pool.run(self._lexical.remove_document, document_id)
                await self._writer.delete_document(document_id)
        except Exception:  # cleanup must never mask the failure that triggered it
            logger.exception(
                "could not discard the chunks of a failed ingest", extra={"event": "ingest_cleanup_failed"}
            )

    def _parse_and_chunk(self, parser: DocumentParser, data: bytes, name: str, document_id: str) -> list[Chunk]:
        """Runs in a worker thread."""
        try:
            parsed = parser.parse(data, name=name)
        except RagError:
            raise
        except Exception as exc:
            raise DocumentParseFailed() from exc
        # Every chunk is at most ``max_chars`` long, so text that cannot fit the chunk limit is refused before the
        # (long) chunking pass rather than after it.
        non_blank = len(parsed.text) - sum(parsed.text.count(c) for c in " \t\r\n")
        if non_blank > self._limits.max_chunks_per_document * self._chunker.max_chars:
            raise LimitExceeded(details={"max_chunks_per_document": self._limits.max_chunks_per_document})
        chunks = self._chunker.chunk(parsed, document_id=document_id, document_name=name)
        if not chunks:
            raise DocumentEmpty()
        if len(chunks) > self._limits.max_chunks_per_document:
            raise LimitExceeded(details={"max_chunks_per_document": self._limits.max_chunks_per_document})
        return chunks

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

    async def _embed(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        batch = self._limits.embed_batch_size
        slots = asyncio.Semaphore(self._limits.max_concurrent_embed_batches)
        texts = [chunk.text for chunk in chunks]

        async def run(start: int) -> list[list[float]]:
            async with slots:
                return await self._embedder.embed_documents(texts[start : start + batch])

        tasks = [asyncio.create_task(run(i)) for i in range(0, len(texts), batch)]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in tasks:  # first failure (or an outer cancellation): the rest must stop calling the provider
                if task not in done:
                    task.cancel()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            if not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
        vectors = [vector for task in tasks for vector in task.result()]
        if len(vectors) != len(chunks) or any(len(v) != self._embedder.dimension for v in vectors):
            raise EmbeddingFailed("the embedder returned the wrong number or size of vectors")
        return vectors

    def _reindex_lexical(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        assert self._lexical is not None
        self._lexical.remove_document(document_id)
        self._lexical.add(chunks)


def _require_known_shape(document_id: str) -> None:
    """An id this service could never have issued names no document: answer without a store round trip or a lock."""
    if not _DOCUMENT_ID.fullmatch(document_id):
        raise DocumentNotFound()


def _annotate(chunks: Sequence[Chunk], *, sha: str, extension: str, index_version: str) -> list[Chunk]:
    extra: dict[str, str | int | float | bool] = {
        "content_sha256": sha,
        "content_type": extension,
        "chunk_count": len(chunks),
        "index_version": index_version,
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
