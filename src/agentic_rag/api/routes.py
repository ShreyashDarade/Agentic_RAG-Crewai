"""HTTP routes. Each one parses, calls the service, and returns: no business logic lives here (ADR-0001)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from agentic_rag.api.auth import require_api_key
from agentic_rag.api.metrics import CONTENT_TYPE_LATEST
from agentic_rag.api.ops import check_readiness
from agentic_rag.application import Service
from agentic_rag.contracts import (
    DeleteResult,
    DocumentInfo,
    DocumentList,
    IngestResult,
    ProblemDetails,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    SearchRequest,
    SearchResponse,
)
from agentic_rag.contracts.models import MAX_PAGE_SIZE
from agentic_rag.errors import ConfigurationError, PayloadTooLarge

__all__ = ["health_router", "ops_router", "v1_router"]

_PROBLEMS: dict[int | str, dict[str, Any]] = {
    code: {"model": ProblemDetails, "description": text}
    for code, text in {
        401: "Missing or invalid API key",
        422: "Invalid request or limit exceeded",
        502: "A dependency failed",
        503: "Overloaded or a dependency is unavailable",
        504: "Deadline exceeded",
    }.items()
}


def get_service(request: Request) -> Service:
    service: Service | None = getattr(request.app.state, "service", None)
    if service is None:
        raise ConfigurationError("the service is not initialised")
    return service


ServiceDep = Annotated[Service, Depends(get_service)]

v1_router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)], responses=_PROBLEMS)
health_router = APIRouter()
ops_router = APIRouter(dependencies=[Depends(require_api_key)])


@v1_router.post("/query", operation_id="query", tags=["query"])
async def query(body: QueryRequest, service: ServiceDep) -> QueryResponse:
    return await service.query(body)


@v1_router.post("/search", operation_id="search", tags=["query"])
async def search(body: SearchRequest, service: ServiceDep) -> SearchResponse:
    return await service.search(body)


@v1_router.post("/documents", operation_id="ingest_document", tags=["documents"], responses={413: _PROBLEMS[422]})
async def ingest_document(
    request: Request, response: Response, file: Annotated[UploadFile, File()], service: ServiceDep
) -> IngestResult:
    data = await _read_limited(file, request.app.state.config.max_upload_bytes)
    result = await service.ingest_document(file.filename or "", data)
    response.status_code = 201 if result.created else 200
    return result


@v1_router.get("/documents", operation_id="list_documents", tags=["documents"])
async def list_documents(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    page_token: str | None = None,
) -> DocumentList:
    return await service.list_documents(limit=limit, page_token=page_token)


@v1_router.get("/documents/{document_id}", operation_id="get_document", tags=["documents"])
async def get_document(document_id: str, service: ServiceDep) -> DocumentInfo:
    return await service.get_document(document_id)


@v1_router.delete("/documents/{document_id}", operation_id="delete_document", tags=["documents"])
async def delete_document(document_id: str, service: ServiceDep) -> DeleteResult:
    return await service.delete_document(document_id)


@health_router.get("/healthz", operation_id="healthz", tags=["operations"])
async def healthz() -> dict[str, str]:
    """Liveness: the process answers. Checks no dependency."""
    return {"status": "ok"}


@health_router.get("/readyz", operation_id="readyz", tags=["operations"], responses={503: {"model": ReadyResponse}})
async def readyz(request: Request, service: ServiceDep) -> Response:
    """Readiness: every dependency is reachable and the server is not draining. Fails honestly."""
    report = await check_readiness(request, service)
    return JSONResponse(report.model_dump(), status_code=200 if report.ready else 503)


@ops_router.get("/metrics", operation_id="metrics", tags=["operations"], response_class=Response)
async def metrics(request: Request, service: ServiceDep) -> Response:
    """Prometheus exposition. Readiness is re-evaluated first so the gauges match what /readyz says."""
    await check_readiness(request, service)
    return Response(request.app.state.metrics.render(), media_type=CONTENT_TYPE_LATEST)


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLarge(details={"max_bytes": max_bytes})
        chunks.append(chunk)
    return b"".join(chunks)
