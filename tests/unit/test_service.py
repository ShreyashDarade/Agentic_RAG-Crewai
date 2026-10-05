"""The application service over fakes: pure logic and failure injection, no network."""

from __future__ import annotations

import json

import pytest

from agentic_rag.adapters.chunker_recursive import RecursiveChunker
from agentic_rag.adapters.lexical_bm25 import Bm25Index
from agentic_rag.adapters.parser_text import TextParser
from agentic_rag.adapters.pipeline_direct import NO_INFORMATION, DirectPipeline
from agentic_rag.application import Limits, RetrievalConfig, Service
from agentic_rag.application.retrieval import HybridRetriever
from agentic_rag.contracts import QueryRequest, SearchRequest
from agentic_rag.errors import (
    DeadlineExceeded,
    DocumentNotFound,
    EmbeddingFailed,
    IndexIncompatible,
    LimitExceeded,
    ModelOutputInvalid,
    PayloadTooLarge,
    UnsupportedFileType,
    ValidationFailed,
)
from agentic_rag.testing import FakeChatModel, FakeEmbedder, FakeVectorStore

TEXT = (
    b"Apples are red fruit grown on trees in many orchards around the world.\n\n"
    b"Bananas are yellow and grow in large bunches on tall tropical plants.\n\n"
    b"The sky is blue because of how sunlight scatters in the atmosphere."
)


def build(
    chat: FakeChatModel | None = None,
    *,
    limits: Limits | None = None,
    embedder: FakeEmbedder | None = None,
    store: FakeVectorStore | None = None,
):
    store = store or FakeVectorStore()
    emb = embedder or FakeEmbedder(32)
    chat = chat or FakeChatModel(["{}"])
    lexical = Bm25Index()
    cfg = RetrievalConfig()
    service = Service(
        embedder=emb,
        writer=store,
        catalog=store,
        retriever=HybridRetriever(embedder=emb, searcher=store, lexical=lexical, reranker=None, config=cfg),
        lexical=lexical,
        parsers=[TextParser()],
        chunker=RecursiveChunker(max_chars=100, overlap_chars=10),
        pipeline=DirectPipeline(chat),
        limits=limits or Limits(),
        retrieval=cfg,
        health_checks={"vector_store": store},
    )
    return service, store, chat


async def test_ingest_is_idempotent_and_content_addressed() -> None:
    service, store, _ = build()
    first = await service.ingest_document("a.txt", TEXT)
    again = await service.ingest_document("renamed.txt", TEXT)
    assert first.created and not again.created
    assert first.document.id == again.document.id and first.chunks_indexed >= 2
    listed = await service.list_documents()
    assert [d.id for d in listed.items] == [first.document.id]


async def test_a_different_embedding_model_cannot_write_into_an_existing_index() -> None:
    service, store, _ = build()
    await service.ingest_document("a.txt", TEXT)
    other, *_ = build(embedder=FakeEmbedder(16), store=store)
    with pytest.raises(IndexIncompatible):
        await other.ingest_document("b.txt", TEXT + b" more words to make different content")


async def test_query_returns_cited_sources_and_flags_grounding() -> None:
    service, _, chat = build()
    ing = await service.ingest_document("a.txt", TEXT)
    chunk_ids = [h.chunk_id for h in (await service.search(SearchRequest(query="apples red fruit"))).hits]
    chat._replies = [json.dumps({"answer": "Apples are red.", "citations": [chunk_ids[0]]})]
    out = await service.query(QueryRequest(question="what colour are apples?", document_ids=[ing.document.id]))
    assert out.grounded and out.citations == [chunk_ids[0]] and out.pipeline == "direct"
    assert chunk_ids[0] in {s.chunk_id for s in out.sources}


async def test_query_with_nothing_indexed_does_not_call_the_model() -> None:
    service, _, chat = build()
    out = await service.query(QueryRequest(question="anything at all"))
    assert out.answer == NO_INFORMATION and not out.grounded and out.sources == []
    assert chat.calls == []


async def test_a_citation_that_was_not_retrieved_is_rejected() -> None:
    service, _, chat = build()
    await service.ingest_document("a.txt", TEXT)
    chat._replies = [json.dumps({"answer": "x", "citations": ["ch_invented"]})]
    with pytest.raises(ModelOutputInvalid):
        await service.query(QueryRequest(question="apples"))


async def test_delete_removes_chunks_and_lexical_entries() -> None:
    service, _, _ = build()
    ing = await service.ingest_document("a.txt", TEXT)
    deleted = await service.delete_document(ing.document.id)
    assert deleted.chunks_deleted == ing.chunks_indexed
    assert (await service.search(SearchRequest(query="apples"))).hits == []
    with pytest.raises(DocumentNotFound):
        await service.get_document(ing.document.id)
    with pytest.raises(DocumentNotFound):
        await service.delete_document(ing.document.id)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"name": "x.exe", "data": b"MZ"}, UnsupportedFileType),
        ({"name": "x.txt", "data": b""}, Exception),
        ({"name": "../../etc/passwd.txt", "data": b" "}, Exception),
    ],
)
async def test_bad_uploads_raise_typed_errors(kwargs: dict[str, object], error: type[Exception]) -> None:
    service, _, _ = build()
    with pytest.raises(error) as exc:
        await service.ingest_document(kwargs["name"], kwargs["data"])  # type: ignore[arg-type]
    assert hasattr(exc.value, "code")


async def test_limits_are_enforced_with_typed_rejections() -> None:
    service, _, _ = build(limits=Limits(max_upload_bytes=10, max_question_chars=5, max_top_k=2))
    with pytest.raises(PayloadTooLarge):
        await service.ingest_document("a.txt", b"x" * 11)
    with pytest.raises(LimitExceeded):
        await service.query(QueryRequest(question="too long question"))
    with pytest.raises(LimitExceeded):
        await service.search(SearchRequest(query="ok", top_k=3))
    with pytest.raises(ValidationFailed):
        await service.list_documents(page_token="not-a-token")


async def test_embedding_failure_is_typed_and_nothing_is_written() -> None:
    service, store, _ = build(embedder=FakeEmbedder(32, fail=True))
    with pytest.raises(EmbeddingFailed):
        await service.ingest_document("a.txt", TEXT)
    assert await store.list_documents(offset=0, limit=10) == []


async def test_deadline_is_a_typed_error() -> None:
    import asyncio

    class SlowChat(FakeChatModel):
        async def complete(self, *a, **k):  # type: ignore[no-untyped-def]
            await asyncio.sleep(1)
            return await super().complete(*a, **k)

    service, _, _ = build(SlowChat(["{}"]), limits=Limits(request_deadline_seconds=0.05))
    await service.ingest_document("a.txt", TEXT)
    with pytest.raises(DeadlineExceeded):
        await service.query(QueryRequest(question="apples"))


async def test_readiness_reports_each_dependency_without_raw_text() -> None:
    service, store, _ = build()
    assert (await service.ready()).ready
    store._unavailable = True
    state = await service.ready()
    assert not state.ready and state.checks == {"vector_store": "unavailable", "index": "unavailable"}
