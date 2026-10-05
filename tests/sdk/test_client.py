"""Sync bridge, configuration safety, live-socket HTTP, and the thin-install guarantee."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import threading
import time
import warnings
from typing import Any

import httpx
import pytest

from agentic_rag import AsyncClient, Client
from agentic_rag.errors import (
    AuthenticationFailed,
    ConnectionFailed,
    DocumentNotFound,
    RagError,
    UsageError,
    ValidationFailed,
)
from tests.conftest import API_KEY
from tests.sdk.conftest import LiveServer
from tests.unit.test_service import TEXT


def _bridge_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "agentic-rag-client")


def test_sync_client_over_a_real_socket(live_server: LiveServer) -> None:
    with Client.http(live_server.url, api_key=API_KEY) as c:
        doc = c.ingest_document("a.txt", TEXT).document
        assert c.search("apples", top_k=2).hits[0].document_id == doc.id
        assert [d.id for d in c.iter_documents(page_size=1)] == [doc.id]
        assert c.ready().ready is True
        with pytest.raises(DocumentNotFound):
            c.get_document("doc_missing")
        with pytest.raises(ValidationFailed):
            c.query("")


def test_sync_embedded_client_matches(engine) -> None:  # type: ignore[no-untyped-def]
    service, *_ = engine
    with Client.embedded(service=service) as c:
        doc = c.ingest_document("a.txt", TEXT).document
        assert c.get_document(doc.id).name == "a.txt"
        with pytest.raises(DocumentNotFound):
            c.get_document("doc_missing")


def test_a_wrong_key_is_an_authentication_error(live_server: LiveServer) -> None:
    with Client.http(live_server.url, api_key="wrong-key-wrong-key") as c, pytest.raises(AuthenticationFailed) as err:
        c.list_documents()
    assert err.value.request_id


def test_an_unreachable_server_is_a_typed_error_after_bounded_retries() -> None:
    from agentic_rag.sdk._retry import RetryPolicy

    async def no_sleep(_: float) -> None:
        return None

    client = Client.http("http://127.0.0.1:1", retry=RetryPolicy(max_retries=2, sleep=no_sleep), timeout=5)
    with client as c, pytest.raises(ConnectionFailed):
        c.list_documents()


async def test_a_blocking_call_inside_a_running_loop_is_a_usage_error_not_a_stall(engine) -> None:  # type: ignore[no-untyped-def]
    service, *_ = engine
    c = Client.embedded(service=service)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no "coroutine was never awaited"
            started = time.monotonic()
            with pytest.raises(UsageError, match="running event loop"):
                c.list_documents()
            assert time.monotonic() - started < 1.0
    finally:
        c.close()


def test_close_is_idempotent_cancels_in_flight_calls_and_blocks_new_ones() -> None:
    entered = threading.Event()

    class Hanging:
        async def list_documents(self, **_: Any) -> Any:
            entered.set()
            await asyncio.sleep(3600)

        async def aclose(self) -> None:
            return None

    before = _bridge_threads()
    c = Client(Hanging())  # type: ignore[arg-type]
    outcome: list[BaseException] = []

    def call() -> None:
        try:
            c.list_documents()
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion below
            outcome.append(exc)

    worker = threading.Thread(target=call)
    worker.start()
    assert entered.wait(5)
    c.close()
    worker.join(5)
    assert not worker.is_alive()
    assert isinstance(outcome[0], UsageError)
    c.close()  # a second close is a no-op
    with pytest.raises(UsageError, match="closed"):
        c.list_documents()
    time.sleep(0.2)
    assert _bridge_threads() == before


def test_a_failed_constructor_leaks_no_thread() -> None:
    before = _bridge_threads()
    for bad in ("ftp://example.com", "not a url", ""):
        with pytest.raises(UsageError):
            Client.http(bad)
    with pytest.raises(UsageError):
        Client.embedded(settings=object())  # type: ignore[arg-type]
    assert _bridge_threads() == before


def test_conflicting_or_unsafe_configuration_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTIC_RAG_API_KEY", raising=False)
    with pytest.raises(UsageError, match="plain http"):
        AsyncClient.http("http://rag.example.com", api_key="k" * 20)
    AsyncClient.http("http://rag.example.com", api_key="k" * 20, allow_insecure=True)
    AsyncClient.http("https://rag.example.com", api_key="k" * 20)
    AsyncClient.http("http://localhost:8000", api_key="k" * 20)
    monkeypatch.setenv("AGENTIC_RAG_API_KEY", "other-key-other-key")
    with pytest.raises(UsageError, match="conflicts"):
        AsyncClient.http("https://x.example.com", api_key="k" * 20)
    monkeypatch.delenv("AGENTIC_RAG_API_KEY")
    with pytest.raises(UsageError, match="verify"):
        AsyncClient.http(
            "https://x.example.com", http_client=httpx.AsyncClient(base_url="https://x.example.com"), verify=False
        )
    with pytest.raises(UsageError, match="timeouts"):
        AsyncClient.http("https://x.example.com", timeout=0)
    with pytest.raises(UsageError, match="not both"):
        AsyncClient.embedded(settings=object(), service=object())  # type: ignore[arg-type]


def test_the_thin_client_imports_nothing_from_the_engine() -> None:
    """With every engine library made unimportable, the client still imports and works against a mock server."""
    program = textwrap.dedent(
        """
        import sys

        BLOCKED = {"fastapi", "starlette", "uvicorn", "pymilvus", "openai", "crewai", "torch", "bs4", "lxml",
                   "pymupdf", "fitz", "docx", "pydantic_settings", "numpy", "chromadb", "qdrant_client"}

        for name in BLOCKED:
            sys.modules[name] = None  # exactly what a missing package looks like to the import system

        import httpx
        import agentic_rag
        from agentic_rag import Client, AsyncClient
        from agentic_rag.errors import RagError
        from agentic_rag.models import QueryRequest
        from agentic_rag.testing import FakeEmbedder
        loaded = {m for m, mod in sys.modules.items() if mod is not None and m.split(".")[0] in BLOCKED}
        assert not loaded, loaded
        assert not any(m.startswith(("agentic_rag.container", "agentic_rag.config", "agentic_rag.adapters", "agentic_rag.api")) for m in sys.modules)
        assert "Settings" not in agentic_rag.__all__ and "build_container" not in agentic_rag.__all__
        try:
            agentic_rag.Settings
        except ImportError as exc:
            assert "agentic-rag[engine]" in str(exc)
        else:
            raise SystemExit("Settings should need the engine")
        try:
            Client.embedded()
        except ImportError as exc:
            assert "agentic-rag[engine]" in str(exc)
        else:
            raise SystemExit("embedded() should need the engine")
        def handler(request):
            return httpx.Response(200, json={"ready": True, "checks": {}})
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://localhost")
        with Client.http("http://localhost", http_client=http) as c:
            assert c.ready().ready is True
        print("thin-ok")
        """
    )
    # -I would drop PYTHONPATH; the editable install provides agentic_rag, so plain -c is correct here.
    proc = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "thin-ok" in proc.stdout


def test_every_ragerror_raised_by_the_client_carries_a_code() -> None:
    for exc in (UsageError("x"), ConnectionFailed(), ValidationFailed()):
        assert isinstance(exc, RagError)
        assert exc.code
