"""The HTTP backend (httpx is imported only here and in ``client``).

Retry matrix (framework section 8):

* never sent (connect errors, no free connection) and refused before work (429, 503): retried for every call;
* ambiguous (connection reset after the request was written, 408, 502, 504): retried only for idempotent calls;
* a read timeout is never retried; other statuses are not retried;
* a DELETE that is retried and then sees ``DOCUMENT_NOT_FOUND`` means the first attempt worked.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ValidationError

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
from agentic_rag.errors import (
    ClientTimeout,
    ConnectionFailed,
    DocumentNotFound,
    InvalidResponse,
    RagError,
    RagStatusError,
    UsageError,
    error_from_problem,
)
from agentic_rag.sdk._retry import RetryPolicy, parse_retry_after

__all__ = ["HttpBackend"]

M = TypeVar("M", bound=BaseModel)

_NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_AMBIGUOUS = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.CloseError,
)
_REFUSED_BEFORE_WORK = {429, 503}
_AMBIGUOUS_STATUS = {408, 502, 504}


class HttpBackend:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        owns_client: bool,
        api_key: str | None,
        timeout: float,
        connect_timeout: float,
        retry: RetryPolicy,
    ) -> None:
        self._client = client
        self._owns = owns_client
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._retry = retry

    # -- backend methods -------------------------------------------------------------------------

    async def query(self, request: QueryRequest) -> QueryResponse:
        body = request.model_dump(mode="json", exclude_none=True)
        return self._parse(QueryResponse, await self._send("POST", "/v1/query", idempotent=True, json_body=body))

    async def search(self, request: SearchRequest) -> SearchResponse:
        body = request.model_dump(mode="json", exclude_none=True)
        return self._parse(SearchResponse, await self._send("POST", "/v1/search", idempotent=True, json_body=body))

    async def ingest_document(self, name: str, data: bytes) -> IngestResult:
        # Ingest is idempotent by content addressing, so ambiguous failures may be retried.
        response = await self._send(
            "POST", "/v1/documents", idempotent=True, files={"file": (name, data, "application/octet-stream")}
        )
        return self._parse(IngestResult, response)

    async def list_documents(self, *, limit: int, page_token: str | None) -> DocumentList:
        params: dict[str, Any] = {"limit": limit}
        if page_token is not None:
            params["page_token"] = page_token
        return self._parse(DocumentList, await self._send("GET", "/v1/documents", idempotent=True, params=params))

    async def get_document(self, document_id: str) -> DocumentInfo:
        return self._parse(DocumentInfo, await self._send("GET", _doc_path(document_id), idempotent=True))

    async def delete_document(self, document_id: str) -> DeleteResult:
        try:
            response = await self._send(
                "DELETE", _doc_path(document_id), idempotent=True, treat_retried_404_as_done=True
            )
        except _RetriedDeleteSucceeded:
            return DeleteResult(document_id=document_id, chunks_deleted=0)
        return self._parse(DeleteResult, response)

    async def ready(self) -> ReadyResponse:
        response = await self._send("GET", "/readyz", idempotent=True, accept=(200, 503))
        return self._parse(ReadyResponse, response)

    async def aclose(self) -> None:
        if self._owns:
            await self._client.aclose()

    # -- machinery -------------------------------------------------------------------------------

    @staticmethod
    def _parse(model: type[M], response: httpx.Response) -> M:
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise InvalidResponse(request_id=response.headers.get("x-request-id")) from exc

    async def _send(
        self,
        method: str,
        path: str,
        *,
        idempotent: bool,
        json_body: Mapping[str, Any] | None = None,
        files: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        accept: tuple[int, ...] = (200, 201),
        treat_retried_404_as_done: bool = False,
    ) -> httpx.Response:
        policy = self._retry
        deadline = policy.monotonic() + self._timeout
        request_id = uuid.uuid4().hex  # one id per logical call, reused by every attempt
        headers = {**self._headers, "X-Request-ID": request_id}
        attempt = 0
        while True:
            remaining = deadline - policy.monotonic()
            if remaining <= 0:
                raise ClientTimeout(request_id=request_id)
            failure: RagError | None = None
            delay: float | None = None
            retry_ok = False
            try:
                response = await self._client.request(
                    method,
                    path,
                    headers=headers,
                    json=json_body,
                    files=files,
                    params=params,
                    timeout=httpx.Timeout(remaining, connect=min(self._connect_timeout, remaining)),
                )
            except _NEVER_SENT as exc:
                failure, retry_ok = ConnectionFailed(request_id=request_id), True
                failure.__cause__ = exc
            except httpx.ReadTimeout as exc:
                raise ClientTimeout(request_id=request_id) from exc  # never retried: the work may be running
            except _AMBIGUOUS as exc:
                failure, retry_ok = ConnectionFailed(request_id=request_id), idempotent
                failure.__cause__ = exc
            except httpx.TimeoutException as exc:
                raise ClientTimeout(request_id=request_id) from exc
            except (httpx.InvalidURL, httpx.UnsupportedProtocol) as exc:
                raise UsageError("the base URL is not usable") from exc
            except httpx.HTTPError as exc:
                raise ConnectionFailed(request_id=request_id) from exc
            else:
                if response.status_code in accept:
                    return response
                failure = _error(response, request_id)
                if treat_retried_404_as_done and attempt > 0 and isinstance(failure, DocumentNotFound):
                    raise _RetriedDeleteSucceeded
                if response.status_code in _REFUSED_BEFORE_WORK:
                    retry_ok = True
                elif response.status_code in _AMBIGUOUS_STATUS:
                    retry_ok = idempotent
                delay = parse_retry_after(response.headers.get("retry-after"))
            assert failure is not None
            if not retry_ok or attempt >= policy.max_retries or not policy.bucket.try_take():
                raise failure
            wait = policy.backoff(attempt) if delay is None else delay
            if (delay is not None and delay > policy.retry_after_cap) or wait >= deadline - policy.monotonic():
                raise failure  # the server asked for longer than we are willing or able to wait
            await policy.sleep(wait)
            attempt += 1


class _RetriedDeleteSucceeded(Exception):
    pass


def _doc_path(document_id: str) -> str:
    return "/v1/documents/" + quote(document_id, safe="")


def _error(response: httpx.Response, request_id: str) -> RagError:
    header_id = response.headers.get("x-request-id") or request_id
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError):
        body = None
    if isinstance(body, dict) and "code" in body:
        error = error_from_problem(body, status=response.status_code)
        error.request_id = error.request_id or header_id
        return error
    return RagStatusError(server_code=f"HTTP_{response.status_code}", status=response.status_code, request_id=header_id)
