"""The public SDK: one facade, two transports, identical behaviour (ADR-0004).

``AsyncClient`` is the implementation; ``Client`` blocks over one background event loop. Both come in
two flavours with the same methods, models and errors::

    client = Client.http("https://rag.example.com", api_key="...")   # remote
    client = Client.embedded()                                       # in-process; needs agentic-rag[engine]

Guarantees: failed calls raise a :class:`agentic_rag.errors.RagError` subclass with a stable ``code``;
retries follow the matrix in docs/framework.md section 8; unknown fields in responses are ignored.
Not guaranteed: streaming (the API has none), ordering of concurrent calls.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from agentic_rag.contracts import (
    DeleteResult,
    DocumentInfo,
    DocumentList,
    IngestResult,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    SearchRequest,
    SearchResponse,
)
from agentic_rag.errors import UsageError, ValidationFailed
from agentic_rag.sdk._backend import Backend
from agentic_rag.sdk._http import HttpBackend
from agentic_rag.sdk._retry import RetryPolicy
from agentic_rag.sdk._sync import LoopThread

if TYPE_CHECKING:
    from agentic_rag.application import Service

__all__ = ["AsyncClient", "Client", "RetryPolicy"]

T = TypeVar("T")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
ENV_API_KEY = "AGENTIC_RAG_API_KEY"


def _validated(build: Callable[[], T]) -> T:
    """Build a request model; invalid input raises ``ValidationFailed`` exactly like the server does."""
    try:
        return build()
    except ValidationError as exc:
        errors = [{"loc": [str(p) for p in e["loc"]], "msg": str(e["msg"])} for e in exc.errors()]
        raise ValidationFailed(details={"errors": errors}) from None


def _read_upload(data: bytes | bytearray | memoryview | BinaryIO | Path) -> bytes:
    """Bytes of the upload. A file object is read from its current position to the end."""
    if isinstance(data, bytes | bytearray | memoryview):
        return bytes(data)
    if isinstance(data, Path):
        return data.read_bytes()
    return data.read()


def _build_http_backend(
    base_url: str,
    api_key: str | None,
    timeout: float,
    connect_timeout: float,
    retry: RetryPolicy | None,
    http_client: httpx.AsyncClient | None,
    verify: bool | str,
    allow_insecure: bool,
) -> HttpBackend:
    env_key = os.environ.get(ENV_API_KEY)
    if api_key and env_key and api_key != env_key:
        raise UsageError(f"api_key conflicts with {ENV_API_KEY}; pass only one")
    key = api_key or env_key or None
    if http_client is not None and not str(http_client.base_url):
        raise UsageError("a supplied http_client must have base_url set")
    # A supplied client sends to its own base URL, so that is the one the plain-http check must look at.
    parts = urlsplit(str(http_client.base_url) if http_client is not None else base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise UsageError("base_url must be an http(s) URL")
    if key and parts.scheme == "http" and parts.hostname not in _LOCAL_HOSTS and not allow_insecure:
        raise UsageError("refusing to send an API key over plain http to a non-local host (allow_insecure=True)")
    if http_client is not None and verify is not True:
        raise UsageError("pass either http_client or verify, not both: verify would be ignored")
    if timeout <= 0 or connect_timeout <= 0:
        raise UsageError("timeouts must be positive")
    owns = http_client is None
    client = http_client or httpx.AsyncClient(base_url=base_url.rstrip("/"), verify=verify)
    return HttpBackend(
        client=client,
        owns_client=owns,
        api_key=key,
        timeout=timeout,
        connect_timeout=connect_timeout,
        retry=retry or RetryPolicy(),
    )


def _build_embedded_backend(settings: object | None, service: Service | None) -> Backend:
    try:
        module = importlib.import_module("agentic_rag.embedded")
    except ImportError as exc:
        raise ImportError("the embedded client needs the engine: pip install 'agentic-rag[engine]'") from exc
    backend: Backend = module.EmbeddedBackend(settings=settings, service=service)
    return backend


class AsyncClient:
    """Asynchronous client. Use as ``async with`` or call :meth:`aclose`."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    @classmethod
    def http(
        cls,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        connect_timeout: float = 5.0,
        retry: RetryPolicy | None = None,
        http_client: httpx.AsyncClient | None = None,
        verify: bool | str = True,
        allow_insecure: bool = False,
    ) -> AsyncClient:
        """Talk to a server. ``timeout`` is the total deadline across retries (keep it above the server's)."""
        return cls(
            _build_http_backend(base_url, api_key, timeout, connect_timeout, retry, http_client, verify, allow_insecure)
        )

    @classmethod
    def embedded(cls, settings: object | None = None, *, service: Service | None = None) -> AsyncClient:
        """Run the engine in this process. ``settings`` defaults to the environment (``AGENTIC_RAG_*``)."""
        if settings is not None and service is not None:
            raise UsageError("pass settings or service, not both")
        return cls(_build_embedded_backend(settings, service))

    async def query(
        self, question: str, *, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> QueryResponse:
        request = _validated(lambda: QueryRequest(question=question, top_k=top_k, document_ids=document_ids))
        return await self._backend.query(request)

    async def search(
        self, query: str, *, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> SearchResponse:
        request = _validated(lambda: SearchRequest(query=query, top_k=top_k, document_ids=document_ids))
        return await self._backend.search(request)

    async def ingest_document(self, name: str, data: bytes | BinaryIO | Path) -> IngestResult:
        """Index a document. A file object is read from its *current* position, identically on both transports."""
        return await self._backend.ingest_document(name, _read_upload(data))

    async def list_documents(self, *, limit: int = 50, page_token: str | None = None) -> DocumentList:
        return await self._backend.list_documents(limit=limit, page_token=page_token)

    async def iter_documents(self, *, page_size: int = 50) -> AsyncIterator[DocumentInfo]:
        token: str | None = None
        while True:
            page = await self._backend.list_documents(limit=page_size, page_token=token)
            for item in page.items:
                yield item
            token = page.next_page_token
            if token is None:
                return

    async def get_document(self, document_id: str) -> DocumentInfo:
        return await self._backend.get_document(document_id)

    async def delete_document(self, document_id: str) -> DeleteResult:
        return await self._backend.delete_document(document_id)

    async def ready(self) -> ReadyResponse:
        """Dependency readiness. Returns the report even when not ready (it does not raise)."""
        return await self._backend.ready()

    async def aclose(self) -> None:
        await self._backend.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class Client:
    """Blocking client: the same methods as :class:`AsyncClient`, run on one background event loop.

    Calling it from inside a running event loop raises ``UsageError``.
    """

    def __init__(self, backend: Backend) -> None:
        self._async = AsyncClient(backend)
        self._loop = LoopThread()  # started last: a failed constructor leaves no thread behind

    @classmethod
    def http(cls, base_url: str, **options: Any) -> Client:
        return cls(AsyncClient.http(base_url, **options)._backend)

    @classmethod
    def embedded(cls, settings: object | None = None, *, service: Service | None = None) -> Client:
        return cls(AsyncClient.embedded(settings, service=service)._backend)

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        return self._loop.run(coro)

    def query(self, question: str, *, top_k: int | None = None, document_ids: list[str] | None = None) -> QueryResponse:
        return self._run(self._async.query(question, top_k=top_k, document_ids=document_ids))

    def search(self, query: str, *, top_k: int | None = None, document_ids: list[str] | None = None) -> SearchResponse:
        return self._run(self._async.search(query, top_k=top_k, document_ids=document_ids))

    def ingest_document(self, name: str, data: bytes | BinaryIO | Path) -> IngestResult:
        return self._run(self._async.ingest_document(name, data))

    def list_documents(self, *, limit: int = 50, page_token: str | None = None) -> DocumentList:
        return self._run(self._async.list_documents(limit=limit, page_token=page_token))

    def iter_documents(self, *, page_size: int = 50) -> Iterator[DocumentInfo]:
        token: str | None = None
        while True:
            page = self.list_documents(limit=page_size, page_token=token)
            yield from page.items
            token = page.next_page_token
            if token is None:
                return

    def get_document(self, document_id: str) -> DocumentInfo:
        return self._run(self._async.get_document(document_id))

    def delete_document(self, document_id: str) -> DeleteResult:
        return self._run(self._async.delete_document(document_id))

    def ready(self) -> ReadyResponse:
        return self._run(self._async.ready())

    def close(self) -> None:
        """Cancel in-flight calls and release everything. Idempotent."""
        self._loop.close(self._async.aclose())

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
