"""The whole stack on a real Milvus (Lite locally, standalone in CI). Embedder and chat model are the
in-memory fakes: no OpenAI key was available, so the OpenAI path is covered only by conformance tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from agentic_rag.api import ApiConfig, create_app
from agentic_rag.config import load_settings
from agentic_rag.container import Container, build_container, default_registries
from agentic_rag.contracts import QueryRequest, SearchRequest
from agentic_rag.errors import VectorStoreUnavailable
from agentic_rag.testing import FakeChatModel, FakeEmbedder
from tests.conftest import API_KEY
from tests.unit.test_service import TEXT

pytestmark = pytest.mark.milvus


def _env(uri: str, collection: str) -> dict[str, str]:
    return {
        "AGENTIC_RAG_API_KEYS": API_KEY,
        "AGENTIC_RAG_MILVUS_URI": uri,
        "AGENTIC_RAG_MILVUS_COLLECTION": collection,
        "AGENTIC_RAG_EMBEDDER": "fake",
        "AGENTIC_RAG_CHAT_MODEL": "fake",
        "AGENTIC_RAG_EMBEDDING_DIMENSION": "32",
        "AGENTIC_RAG_CHUNK_MAX_CHARS": "100",
        "AGENTIC_RAG_CHUNK_OVERLAP_CHARS": "10",
    }


async def _build(uri: str, collection: str, chat: FakeChatModel | None = None) -> Container:
    regs = default_registries()
    regs.embedder.register("fake", lambda s, ctx: FakeEmbedder(s.embedding_dimension))
    regs.chat_model.register("fake", lambda s, ctx: chat or FakeChatModel(["{}"]))
    return await build_container(load_settings(_env(uri, collection)), regs)


async def _until_ready(container: Container) -> None:
    for _ in range(100):
        if (await container.service.ready()).ready:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"never became ready: {await container.service.ready()}")


@pytest.fixture
async def container(milvus_uri: str, collection_name: str) -> AsyncIterator[Container]:
    c = await _build(milvus_uri, collection_name)
    await _until_ready(c)
    yield c
    await c.aclose()


async def test_ingest_search_query_delete_on_a_real_milvus(container: Container) -> None:
    svc = container.service
    first = await svc.ingest_document("fruit.txt", TEXT)
    assert first.created and first.chunks_indexed >= 2
    assert not (await svc.ingest_document("again.txt", TEXT)).created
    hits = (await svc.search(SearchRequest(query="bananas yellow bunches", top_k=3))).hits
    assert hits
    assert "Bananas" in hits[0].text
    assert hits[0].document_id == first.document.id
    assert (await svc.list_documents()).items[0].chunk_count == first.chunks_indexed
    out = await svc.query(QueryRequest(question="zzzz unrelated", document_ids=["doc_missing"]))
    assert not out.grounded
    deleted = await svc.delete_document(first.document.id)
    assert deleted.chunks_deleted == first.chunks_indexed
    assert (await svc.search(SearchRequest(query="bananas"))).hits == []


async def test_pdf_and_docx_through_the_real_stack(container: Container) -> None:
    import io

    import docx
    import pymupdf

    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Quarterly revenue grew nine percent in the northern region.")
    document = docx.Document()
    document.add_paragraph("Penguins live in the southern hemisphere and eat krill.")
    buffer = io.BytesIO()
    document.save(buffer)
    await container.service.ingest_document("q.pdf", pdf.tobytes())
    await container.service.ingest_document("p.docx", buffer.getvalue())
    pdf_hit = (await container.service.search(SearchRequest(query="revenue northern region", top_k=1))).hits[
        0
    ]
    assert pdf_hit.document_name == "q.pdf"
    assert pdf_hit.page == 1
    docx_hit = (await container.service.search(SearchRequest(query="penguins krill", top_k=1))).hits[0]
    assert docx_hit.document_name == "p.docx"


async def test_restart_rebuilds_the_lexical_index_from_the_store(
    milvus_uri: str, collection_name: str
) -> None:
    first = await _build(milvus_uri, collection_name)
    await _until_ready(first)
    await first.service.ingest_document("fruit.txt", TEXT)
    await first.aclose()
    second = await _build(milvus_uri, collection_name)
    try:
        await _until_ready(second)
        assert (await second.service.ready()).checks == {"vector_store": "ok", "lexical_index": "ok"}
        assert (await second.service.search(SearchRequest(query="tropical plants", top_k=2))).hits
    finally:
        await second.aclose()


async def test_the_http_app_over_the_real_container(container: Container) -> None:
    app = create_app(config=ApiConfig(api_keys=(API_KEY,)), service=container.service)
    headers = {"Authorization": f"Bearer {API_KEY}"}
    with TestClient(app) as client:
        r = client.post("/v1/documents", headers=headers, files={"file": ("f.txt", TEXT, "text/plain")})
        assert r.status_code == 201
        chunk = client.post("/v1/search", headers=headers, json={"query": "sunlight scatters"}).json()[
            "hits"
        ][0]
        assert "sunlight" in chunk["text"]
        assert client.get("/readyz").json()["ready"] is True


async def test_an_unreachable_milvus_is_reported_honestly_not_hidden() -> None:
    c = await _build("http://127.0.0.1:1", "docs")
    try:
        ready = await c.service.ready()
        assert not ready.ready
        assert ready.checks["vector_store"] == "unavailable"
        with pytest.raises(VectorStoreUnavailable):
            await c.service.search(SearchRequest(query="anything"))
        with pytest.raises(VectorStoreUnavailable):
            await c.service.ingest_document("a.txt", TEXT)
    finally:
        await c.aclose()
    assert json.dumps(ready.model_dump())  # serialisable, no raw dependency text


async def test_a_collection_built_with_another_embedding_size_is_refused(
    milvus_uri: str, collection_name: str
) -> None:
    first = await _build(milvus_uri, collection_name)
    await first.service.ingest_document("a.txt", TEXT)
    await first.aclose()
    other = load_settings({**_env(milvus_uri, collection_name), "AGENTIC_RAG_EMBEDDING_DIMENSION": "16"})
    regs = default_registries()
    regs.embedder.register("fake", lambda s, ctx: FakeEmbedder(s.embedding_dimension))
    regs.chat_model.register("fake", lambda s, ctx: FakeChatModel(["{}"]))
    second = await build_container(other, regs)
    try:
        from agentic_rag.errors import IndexIncompatible

        with pytest.raises(IndexIncompatible):
            await second.service.ingest_document("b.txt", TEXT + b" extra content so the hash differs")
    finally:
        await second.aclose()
