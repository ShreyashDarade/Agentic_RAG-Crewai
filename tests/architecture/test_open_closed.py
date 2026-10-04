"""G10 / SOLID-O: a component is added by registering it; the core is not edited."""

from __future__ import annotations

from agentic_rag.config import load_settings
from agentic_rag.container import build_container
from agentic_rag.contracts import QueryRequest
from tests.plugin_full import SHARED_STORE
from tests.unit.test_service import TEXT

ENV = {
    "AGENTIC_RAG_PLUGINS": "tests.plugin_full",
    "AGENTIC_RAG_API_KEYS": "k" * 20,
    "AGENTIC_RAG_VECTOR_STORE": "memory",
    "AGENTIC_RAG_EMBEDDER": "hashing",
    "AGENTIC_RAG_CHAT_MODEL": "scripted",
    "AGENTIC_RAG_ANSWER_PIPELINE": "shouting",
    "AGENTIC_RAG_EMBEDDING_DIMENSION": "32",
    "AGENTIC_RAG_CHUNK_MAX_CHARS": "100",
    "AGENTIC_RAG_CHUNK_OVERLAP_CHARS": "10",
}


async def test_a_plugin_adds_store_embedder_model_and_pipeline_without_touching_the_core() -> None:
    container = await build_container(load_settings(ENV))
    try:
        ingested = await container.service.ingest_document("a.txt", TEXT)
        out = await container.service.query(QueryRequest(question="what are apples"))
        assert out.pipeline == "shouting"
        assert out.answer == "WHAT ARE APPLES"
        assert out.grounded
        assert (await SHARED_STORE.get_document(ingested.document.id)) is not None
    finally:
        await container.aclose()
