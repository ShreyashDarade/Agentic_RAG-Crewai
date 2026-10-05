"""The embedded backend: the engine in this process, behind the same facade as HTTP (ADR-0004).

It builds the container lazily on first use (inside whichever event loop runs the call) and passes
every result through the contract models' JSON form, so callers get the same types as over HTTP.
"""

from __future__ import annotations

import asyncio
from typing import TypeVar

from pydantic import BaseModel

from agentic_rag.application import Service
from agentic_rag.config import Settings, load_settings
from agentic_rag.container import Container, build_container
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
from agentic_rag.errors import UsageError

__all__ = ["EmbeddedBackend"]

M = TypeVar("M", bound=BaseModel)


def _same_types_as_http(model: type[M], value: BaseModel) -> M:
    return model.model_validate(value.model_dump(mode="json"))


class EmbeddedBackend:
    def __init__(self, *, settings: object | None = None, service: Service | None = None) -> None:
        if settings is not None and not isinstance(settings, Settings):
            raise UsageError("settings must be agentic_rag.config.Settings")
        self._settings = settings
        self._service = service
        self._container: Container | None = None
        self._lock: asyncio.Lock | None = None

    async def _svc(self) -> Service:
        if self._service is not None:
            return self._service
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._service is None:
                settings = self._settings if isinstance(self._settings, Settings) else load_settings()
                self._container = await build_container(settings)
                self._service = self._container.service
        return self._service

    async def query(self, request: QueryRequest) -> QueryResponse:
        return _same_types_as_http(QueryResponse, await (await self._svc()).query(request))

    async def search(self, request: SearchRequest) -> SearchResponse:
        return _same_types_as_http(SearchResponse, await (await self._svc()).search(request))

    async def ingest_document(self, name: str, data: bytes) -> IngestResult:
        return _same_types_as_http(IngestResult, await (await self._svc()).ingest_document(name, data))

    async def list_documents(self, *, limit: int, page_token: str | None) -> DocumentList:
        result = await (await self._svc()).list_documents(limit=limit, page_token=page_token)
        return _same_types_as_http(DocumentList, result)

    async def get_document(self, document_id: str) -> DocumentInfo:
        return _same_types_as_http(DocumentInfo, await (await self._svc()).get_document(document_id))

    async def delete_document(self, document_id: str) -> DeleteResult:
        return _same_types_as_http(DeleteResult, await (await self._svc()).delete_document(document_id))

    async def ready(self) -> ReadyResponse:
        return _same_types_as_http(ReadyResponse, await (await self._svc()).ready())

    async def aclose(self) -> None:
        if self._container is not None:
            container, self._container = self._container, None
            await container.aclose()
