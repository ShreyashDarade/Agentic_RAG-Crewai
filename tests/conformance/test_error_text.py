"""G12/security: text a dependency produces never reaches a caller, through every adapter's own error path."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from agentic_rag.errors import RagError
from tests.stub_openai import StubOpenAI

SECRET = "SECRET-DETAIL"


@pytest.fixture(scope="module")
def stub() -> Iterator[StubOpenAI]:
    server = StubOpenAI().start()
    yield server
    server.stop()


def _public(exc: RagError) -> str:
    """Everything a caller can see of an error: its message, its detail, and the problem+json body."""
    return " ".join([str(exc), exc.detail or "", json.dumps(exc.to_problem())])


async def _provider_calls(stub: StubOpenAI) -> dict[str, Any]:
    from agentic_rag.adapters.openai import OpenAIChatModel, OpenAIEmbedder, create_client
    from agentic_rag.ports import ChatMessage

    client = create_client(api_key="sk-test", base_url=stub.url, timeout_seconds=5)
    calls: dict[str, Any] = {
        "openai-embedder": lambda: OpenAIEmbedder(client, dimension=8).embed_documents(["x"]),
        "openai-chat": lambda: OpenAIChatModel(client).complete([ChatMessage("user", "hi")], max_tokens=5),
    }
    try:
        from agentic_rag.adapters.crewai import CrewAIChatModel, CrewAIEmbedder
    except ImportError:
        return calls
    calls["crewai-chat"] = lambda: CrewAIChatModel("openai/gpt-4o-mini", api_key="sk-test", base_url=stub.url).complete(
        [ChatMessage("user", "hi")], max_tokens=5
    )
    calls["crewai-embedder"] = lambda: CrewAIEmbedder(
        "openai", {"api_key": "sk-test", "api_base": stub.url, "model_name": "text-embedding-3-small"}, dimension=8
    ).embed_documents(["x"])
    return calls


@pytest.mark.parametrize("mode", ["500", "401"])  # 429 is covered by the provider suites (it waits out Retry-After)
async def test_a_provider_error_body_never_reaches_the_caller(stub: StubOpenAI, mode: str) -> None:
    stub.mode = mode
    seen = 0
    for name, call in (await _provider_calls(stub)).items():
        with pytest.raises(RagError) as err:
            await call()
        assert SECRET not in _public(err.value), f"{name} put provider text into its public error"
        seen += 1
    assert seen >= 2
    stub.mode = "ok"


async def test_a_vector_store_error_never_reaches_the_caller(milvus_uri: str, tmp_path: Any) -> None:
    from pymilvus.exceptions import MilvusException

    from agentic_rag.adapters.milvus import MilvusSettings, MilvusStore

    store = MilvusStore(MilvusSettings(uri=milvus_uri))

    def boom(_client: Any) -> None:
        raise MilvusException(code=1100, message=f"{SECRET} token=abc host=10.1.2.3")

    with pytest.raises(RagError) as err:
        await store._run(boom)
    assert SECRET not in _public(err.value)
    await store.aclose()


async def test_chroma_and_qdrant_errors_never_reach_the_caller(tmp_path: Any) -> None:
    chroma = pytest.importorskip("chromadb")
    qdrant = pytest.importorskip("qdrant_client")
    from chromadb.errors import ChromaError
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    from agentic_rag.adapters.chroma import ChromaSettings, ChromaStore
    from agentic_rag.adapters.qdrant import QdrantSettings, QdrantStore

    assert chroma and qdrant
    chroma_store = ChromaStore(ChromaSettings(path=str(tmp_path / "c")))
    qdrant_store = QdrantStore(QdrantSettings(location=":memory:"))
    failures: list[tuple[Any, Exception]] = [
        (chroma_store, ChromaError(f"{SECRET} password=x")),
        (chroma_store, ValueError(f"could not connect to {SECRET}")),
        (chroma_store, ConnectionError(f"{SECRET} refused")),
        (qdrant_store, UnexpectedResponse(500, f"{SECRET}", f"{SECRET} body".encode(), {})),
        (qdrant_store, ResponseHandlingException(Exception(f"{SECRET} api-key=abc"))),
    ]
    for store, exc in failures:

        def boom(_client: Any, exc: Exception = exc) -> None:
            raise exc

        with pytest.raises(RagError) as err:
            await store._run(boom)
        assert SECRET not in _public(err.value), f"{type(store).__name__} leaked {type(exc).__name__}"
    await chroma_store.aclose()
    await qdrant_store.aclose()
