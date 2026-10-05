"""Review round 1 (engine): deadlines everywhere, an ingest that cannot half-finish silently, bounded work."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Collection, Sequence

import pytest

from agentic_rag.adapters.chunker_recursive import RecursiveChunker
from agentic_rag.adapters.parser_text import TextParser
from agentic_rag.application import Limits
from agentic_rag.contracts import QueryRequest, SearchRequest
from agentic_rag.errors import (
    DeadlineExceeded,
    DocumentNotFound,
    EmbeddingFailed,
    IndexIncompatible,
    UpstreamRateLimited,
    VectorStoreUnavailable,
)
from agentic_rag.ports import Chunk, ParsedDocument
from agentic_rag.testing import FakeEmbedder, FakeVectorStore
from tests.unit.test_service import TEXT, build


class SlowStore(FakeVectorStore):
    """Every call waits for ``delay`` seconds first."""

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__()
        self.delay = delay

    async def get_document(self, document_id: str):  # type: ignore[no-untyped-def]
        await asyncio.sleep(self.delay)
        return await super().get_document(document_id)

    async def list_documents(self, *, offset: int, limit: int):  # type: ignore[no-untyped-def]
        await asyncio.sleep(self.delay)
        return await super().list_documents(offset=offset, limit=limit)

    async def delete_document(self, document_id: str, *, keep_chunk_ids: Collection[str] = ()) -> int:
        await asyncio.sleep(self.delay)
        return await super().delete_document(document_id, keep_chunk_ids=keep_chunk_ids)


class RecordingStore(FakeVectorStore):
    """Records the order of writes and can fail one of them."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.fail_on: str | None = None

    def _maybe_fail(self, event: str) -> None:
        self.events.append(event)
        if self.fail_on == event:
            raise VectorStoreUnavailable()

    async def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, embedding_model: str
    ) -> None:
        self._maybe_fail("head" if any(c.index == 0 for c in chunks) else "body")
        await super().upsert(chunks, vectors, embedding_model=embedding_model)

    async def delete_document(self, document_id: str, *, keep_chunk_ids: Collection[str] = ()) -> int:
        if keep_chunk_ids:
            self._maybe_fail("sweep")
        return await super().delete_document(document_id, keep_chunk_ids=keep_chunk_ids)


# -- every use case has a deadline (review A8 / B3) --------------------------------------------------------------


async def test_get_list_and_delete_obey_the_request_deadline() -> None:
    store = SlowStore()
    service, *_ = build(store=store, limits=Limits(request_deadline_seconds=0.2))
    ing = await service.ingest_document("a.txt", TEXT)
    store.delay = 2.0
    for call in (
        lambda: service.get_document(ing.document.id),
        lambda: service.list_documents(),
        lambda: service.delete_document(ing.document.id),
    ):
        started = time.monotonic()
        with pytest.raises(DeadlineExceeded):
            await call()
        assert time.monotonic() - started < 1.0


# -- ingest order and failure (review A3 / A4) -------------------------------------------------------------------


async def test_the_commit_marker_is_written_last() -> None:
    store = RecordingStore()
    service, *_ = build(store=store)
    await service.ingest_document("a.txt", TEXT)
    assert store.events == ["body", "sweep", "head"]  # write -> sweep -> commit (ADR-0007)


async def test_a_failed_commit_leaves_nothing_searchable_and_a_retry_finishes() -> None:
    store = RecordingStore()
    service, *_ = build(store=store)
    store.fail_on = "head"
    with pytest.raises(VectorStoreUnavailable):
        await service.ingest_document("a.txt", TEXT)
    store.fail_on = None
    assert (await service.search(SearchRequest(query="apples red fruit"))).hits == []  # compensated
    assert (await service.list_documents()).items == []
    retried = await service.ingest_document("a.txt", TEXT)
    assert retried.created
    assert (await service.search(SearchRequest(query="apples red fruit"))).hits


async def test_a_retry_after_a_failed_sweep_does_the_whole_job_not_a_cached_half() -> None:
    store = RecordingStore()
    service, *_ = build(store=store)
    store.fail_on = "sweep"
    with pytest.raises(VectorStoreUnavailable):
        await service.ingest_document("a.txt", TEXT)
    store.fail_on = None
    retried = await service.ingest_document("a.txt", TEXT)
    assert retried.created  # nothing was committed, so this is not a "created=False" echo of a partial state
    assert (await service.search(SearchRequest(query="apples red fruit"))).hits  # the lexical index got it too
    lexical_hits = service._lexical.search("apples", top_k=3)  # type: ignore[union-attr]
    assert lexical_hits


async def test_chunks_left_without_a_catalog_entry_can_still_be_deleted() -> None:
    store = FakeVectorStore()
    service, *_ = build(store=store)
    ing = await service.ingest_document("a.txt", TEXT)
    # simulate an interrupted ingest: the commit marker (chunk 0) is gone, the body is not
    for cid, (chunk, _, _) in list(store._rows.items()):
        if chunk.index == 0:
            del store._rows[cid]
    with pytest.raises(DocumentNotFound):
        await service.get_document(ing.document.id)
    result = await service.delete_document(ing.document.id)
    assert result.chunks_deleted == ing.chunks_indexed - 1
    assert store._rows == {}
    with pytest.raises(DocumentNotFound):  # and now there really is nothing
        await service.delete_document(ing.document.id)


# -- re-ingest replaces; identity covers what the parser sees (review A12 / A13) ---------------------------------


async def test_changing_the_chunking_recipe_makes_a_re_upload_replace_the_old_index() -> None:
    store = FakeVectorStore()
    first, *_ = build(store=store)
    a = await first.ingest_document("a.txt", TEXT)
    second, *_ = build(store=store)
    second._chunker = RecursiveChunker(max_chars=200, overlap_chars=20)
    b = await second.ingest_document("a.txt", TEXT)
    assert b.created and b.document.id == a.document.id
    assert b.chunks_indexed != a.chunks_indexed
    assert len(store._rows) == b.chunks_indexed  # the old chunks were swept
    again = await second.ingest_document("a.txt", TEXT)
    assert not again.created  # same recipe: idempotent


async def test_the_same_bytes_uploaded_as_another_file_type_are_another_document() -> None:
    from agentic_rag.adapters.parser_html import HtmlParser

    store = FakeVectorStore()
    service, *_ = build(store=store)
    service._parsers[".html"] = HtmlParser()
    html = b"<html><body><p>Apples are red fruit grown on trees in many orchards.</p><script>var x=1;</script></body></html>"
    as_text = await service.ingest_document("page.txt", html)
    as_html = await service.ingest_document("page.html", html)
    assert as_html.created and as_html.document.id != as_text.document.id
    assert as_html.document.content_sha256 == as_text.document.content_sha256  # the bytes hash is still reported
    html_text = " ".join(c.text for c, _, _ in store._rows.values() if c.document_id == as_html.document.id)
    assert "var x" not in html_text


# -- ids, locks, filters (review A9 / A14) -----------------------------------------------------------------------


@pytest.mark.parametrize("hostile", ["", ".", "..", "x" * 5000, "doc_" + "g" * 32, "../../etc/passwd", "doc_x\x00"])
async def test_ids_that_cannot_exist_are_not_found_without_touching_the_store_or_the_lock_table(hostile: str) -> None:
    store = RecordingStore()
    service, *_ = build(store=store)
    for call in (service.get_document, service.delete_document):
        with pytest.raises(DocumentNotFound):
            await call(hostile)
    assert service._doc_locks == {}
    assert store.events == []


async def test_the_lock_table_does_not_grow() -> None:
    service, *_ = build()
    ing = await service.ingest_document("a.txt", TEXT)
    await service.ingest_document("a.txt", TEXT)
    await service.delete_document(ing.document.id)
    assert service._doc_locks == {}


async def test_an_empty_document_filter_is_rejected_not_treated_as_search_everything() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SearchRequest(query="apples", document_ids=[])
    with pytest.raises(ValidationError):
        QueryRequest(question="apples", document_ids=[])


# -- the index must match the embedder on the read path too (review A5) ------------------------------------------


class _OtherModel(FakeEmbedder):
    @property
    def model_id(self) -> str:
        return "some-other-model-same-size"


async def test_a_different_embedding_model_of_the_same_size_is_refused_on_search_and_readiness() -> None:
    store = FakeVectorStore()
    first, *_ = build(store=store)
    await first.ingest_document("a.txt", TEXT)
    other, *_ = build(store=store, embedder=_OtherModel(32))
    with pytest.raises(IndexIncompatible):
        await other.search(SearchRequest(query="apples"))
    with pytest.raises(IndexIncompatible):
        await other.query(QueryRequest(question="apples"))
    report = await other.ready()
    assert not report.ready and report.checks["index"] == "incompatible"
    ok = await first.ready()
    assert ok.ready and ok.checks["index"] == "ok"


# -- embedding batches (review A10) ------------------------------------------------------------------------------


class _FailingSecondBatch(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(32)
        self.started = 0
        self.finished = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.started += 1
        order = self.started
        if order == 2:
            raise UpstreamRateLimited()
        await asyncio.sleep(0.3)
        self.finished += 1
        return await super().embed_documents(texts)


async def test_one_failed_embedding_batch_cancels_the_others() -> None:
    embedder = _FailingSecondBatch()
    service, *_ = build(embedder=embedder, limits=Limits(embed_batch_size=1, max_concurrent_embed_batches=3))
    with pytest.raises(UpstreamRateLimited):
        await service.ingest_document("a.txt", TEXT * 3)
    await asyncio.sleep(0.5)
    assert embedder.started < 8  # the batches that had not started never did (there are more than a dozen chunks)
    assert embedder.finished == 0  # and the ones that were running were cancelled, not left calling the provider


# -- bounded blocking work (review A2) ---------------------------------------------------------------------------


class _SlowParser(TextParser):
    def __init__(self) -> None:
        self.running = 0
        self.peak = 0
        self.release = threading.Event()
        self._lock = threading.Lock()

    def parse(self, data: bytes, *, name: str) -> ParsedDocument:
        with self._lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            self.release.wait(10)
            return super().parse(data, name=name)
        finally:
            with self._lock:
                self.running -= 1


async def test_timed_out_ingests_cannot_pile_up_parser_threads() -> None:
    parser = _SlowParser()
    service, *_ = build(limits=Limits(request_deadline_seconds=0.2, max_concurrent_ingests=2))
    service._parsers[".txt"] = parser
    results = await asyncio.gather(
        *(service.ingest_document("a.txt", TEXT + str(i).encode()) for i in range(10)), return_exceptions=True
    )
    assert all(isinstance(r, Exception) for r in results)
    assert parser.peak <= 2  # the slot is held until the thread returns, not until the request is cancelled
    parser.release.set()
    await asyncio.sleep(0.2)
    assert parser.running == 0


async def test_a_text_that_cannot_fit_the_chunk_limit_is_refused_before_chunking() -> None:
    from agentic_rag.errors import LimitExceeded

    service, *_ = build(limits=Limits(max_chunks_per_document=3))
    called = False
    real = service._chunker.chunk

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        return real(*args, **kwargs)

    service._chunker.chunk = spy  # type: ignore[method-assign]
    with pytest.raises(LimitExceeded):
        await service.ingest_document("big.txt", b"word " * 5000)
    assert not called


async def test_embedding_failures_stay_typed() -> None:
    service, *_ = build(embedder=FakeEmbedder(32, fail=True))
    with pytest.raises(EmbeddingFailed):
        await service.ingest_document("a.txt", TEXT)
