"""The same end-to-end body on every vector store the container can build (all real engines, all local)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agentic_rag.config import load_settings
from agentic_rag.container import Container, build_container, default_registries
from agentic_rag.contracts import QueryRequest, SearchRequest
from agentic_rag.errors import IndexIncompatible
from agentic_rag.testing import FakeChatModel, FakeEmbedder
from tests.stub_openai import DIM, StubOpenAI
from tests.unit.test_service import TEXT

BASE = {
    "AGENTIC_RAG_API_KEYS": "k" * 20,
    "AGENTIC_RAG_EMBEDDER": "fake",
    "AGENTIC_RAG_CHAT_MODEL": "fake",
    "AGENTIC_RAG_EMBEDDING_DIMENSION": "32",
    "AGENTIC_RAG_CHUNK_MAX_CHARS": "100",
    "AGENTIC_RAG_CHUNK_OVERLAP_CHARS": "10",
}


def store_env(kind: str, tmp: Path) -> dict[str, str]:
    if kind == "milvus":
        pytest.importorskip("milvus_lite")
        return {"AGENTIC_RAG_VECTOR_STORE": "milvus", "AGENTIC_RAG_MILVUS_URI": str(tmp / "m.db")}
    if kind == "chroma":
        pytest.importorskip("chromadb")
        return {"AGENTIC_RAG_VECTOR_STORE": "chroma", "AGENTIC_RAG_CHROMA_PATH": str(tmp / "chroma")}
    pytest.importorskip("qdrant_client")
    return {"AGENTIC_RAG_VECTOR_STORE": "qdrant", "AGENTIC_RAG_QDRANT_LOCATION": str(tmp / "qdrant")}


async def build(env: dict[str, str], chat: FakeChatModel | None = None) -> Container:
    regs = default_registries()
    regs.embedder.register("fake", lambda s, ctx: FakeEmbedder(s.embedding_dimension))
    regs.chat_model.register("fake", lambda s, ctx: chat or FakeChatModel(['{"answer": "ok", "citations": []}']))
    return await build_container(load_settings({**BASE, **env}), regs)


async def ready(c: Container) -> None:
    for _ in range(200):
        if (await c.service.ready()).ready:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(await c.service.ready())


@pytest.fixture(params=["milvus", "chroma", "qdrant"])
async def stack(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[tuple[str, dict[str, str], Container]]:
    env = store_env(request.param, tmp_path)
    container = await build(env)
    await ready(container)
    yield request.param, env, container
    await container.aclose()


async def test_ingest_search_query_delete(stack) -> None:  # type: ignore[no-untyped-def]
    _, _, c = stack
    svc = c.service
    first = await svc.ingest_document("fruit.txt", TEXT)
    assert first.created
    assert first.chunks_indexed >= 2
    assert not (await svc.ingest_document("again.txt", TEXT)).created
    hits = (await svc.search(SearchRequest(query="bananas yellow bunches", top_k=3))).hits
    assert "Bananas" in hits[0].text
    assert hits[0].document_id == first.document.id
    assert [d.id for d in (await svc.list_documents()).items] == [first.document.id]
    out = await svc.query(QueryRequest(question="what are bananas", document_ids=[first.document.id]))
    assert out.pipeline == "direct"
    assert (await svc.delete_document(first.document.id)).chunks_deleted == first.chunks_indexed
    assert (await svc.search(SearchRequest(query="bananas"))).hits == []


async def test_restart_keeps_the_data_and_rebuilds_the_lexical_index(stack) -> None:  # type: ignore[no-untyped-def]
    kind, env, c = stack
    if kind == "qdrant" and env["AGENTIC_RAG_QDRANT_LOCATION"] == ":memory:":
        pytest.skip("memory store")
    await c.service.ingest_document("fruit.txt", TEXT)
    await c.aclose()
    again = await build(env)
    try:
        await ready(again)
        assert (await again.service.ready()).checks == {"vector_store": "ok", "lexical_index": "ok", "index": "ok"}
        assert (await again.service.search(SearchRequest(query="tropical plants", top_k=2))).hits
    finally:
        await again.aclose()


async def test_another_embedding_size_is_refused_by_every_store(stack) -> None:  # type: ignore[no-untyped-def]
    _, env, c = stack
    await c.service.ingest_document("fruit.txt", TEXT)
    await c.aclose()
    other = await build({**env, "AGENTIC_RAG_EMBEDDING_DIMENSION": "16"})
    try:
        with pytest.raises(IndexIncompatible):
            await other.service.ingest_document("b.txt", TEXT + b" different content so the hash differs")
    finally:
        await other.aclose()


async def test_crewai_providers_and_pipeline_through_the_container(tmp_path: Path) -> None:
    """CrewAI embedder + CrewAI chat model + CrewAI crew, over real sockets to a local OpenAI-compatible stub."""
    pytest.importorskip("crewai")
    pytest.importorskip("qdrant_client")
    stub = StubOpenAI().start()
    try:
        stub.chat_reply = "Thought: done\nFinal Answer: " + json.dumps(
            {"answer": "Bananas are yellow.", "citations": []}
        )
        env = {
            **BASE,
            "AGENTIC_RAG_VECTOR_STORE": "qdrant",
            "AGENTIC_RAG_QDRANT_LOCATION": str(tmp_path / "q"),
            "AGENTIC_RAG_EMBEDDER": "crewai",
            "AGENTIC_RAG_CHAT_MODEL": "crewai",
            "AGENTIC_RAG_ANSWER_PIPELINE": "crewai",
            "AGENTIC_RAG_EMBEDDING_DIMENSION": str(DIM),
            "AGENTIC_RAG_CREWAI_EMBEDDER_PROVIDER": "openai",
            "AGENTIC_RAG_CREWAI_EMBEDDER_MODEL": "text-embedding-3-small",
            "AGENTIC_RAG_CREWAI_EMBEDDER_API_KEY": "sk-test",
            "AGENTIC_RAG_CREWAI_EMBEDDER_BASE_URL": stub.url,
            "AGENTIC_RAG_CREWAI_LLM_MODEL": "openai/gpt-4o-mini",
            "AGENTIC_RAG_CREWAI_LLM_API_KEY": "sk-test",
            "AGENTIC_RAG_CREWAI_LLM_BASE_URL": stub.url,
        }
        container = await build_container(load_settings(env))
        try:
            await ready(container)
            ingested = await container.service.ingest_document("fruit.txt", TEXT)
            assert ingested.document.embedding_model == f"crewai:openai:text-embedding-3-small:{DIM}"
            out = await container.service.query(QueryRequest(question="what colour are bananas?"))
            assert out.pipeline == "crewai"
            assert out.answer == "Bananas are yellow."
            paths = {r["path"] for r in stub.requests}
            assert {"/v1/embeddings", "/v1/chat/completions"} <= paths
            assert all(r["auth"] == "Bearer sk-test" for r in stub.requests)
        finally:
            await container.aclose()
    finally:
        stub.stop()
