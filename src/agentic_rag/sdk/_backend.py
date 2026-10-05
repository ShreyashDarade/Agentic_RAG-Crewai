"""The narrow transport protocol behind the SDK facade (ADR-0004).

A backend only moves requests and results; it never adds behaviour the other backend lacks.
Standard library plus the contracts only.
"""

from __future__ import annotations

from typing import Protocol

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

__all__ = ["Backend"]


class Backend(Protocol):
    async def query(self, request: QueryRequest) -> QueryResponse: ...

    async def search(self, request: SearchRequest) -> SearchResponse: ...

    async def ingest_document(self, name: str, data: bytes) -> IngestResult: ...

    async def list_documents(self, *, limit: int, page_token: str | None) -> DocumentList: ...

    async def get_document(self, document_id: str) -> DocumentInfo: ...

    async def delete_document(self, document_id: str) -> DeleteResult: ...

    async def ready(self) -> ReadyResponse: ...

    async def aclose(self) -> None: ...
