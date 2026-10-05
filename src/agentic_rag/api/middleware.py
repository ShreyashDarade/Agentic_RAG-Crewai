"""Pure-ASGI middleware (``BaseHTTPMiddleware`` would break contextvars and streaming)."""

from __future__ import annotations

import contextvars
import json
import logging
import random
import re
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agentic_rag.api.auth import bearer_token, is_authorized
from agentic_rag.api.config import ApiConfig
from agentic_rag.api.metrics import Metrics
from agentic_rag.errors import AuthenticationFailed, Overloaded, PayloadTooLarge, RagError

__all__ = [
    "BODY_EXCEEDED_SCOPE_KEY",
    "REQUEST_ID_SCOPE_KEY",
    "AuthMiddleware",
    "BodyLimitMiddleware",
    "LoadSheddingMiddleware",
    "RequestContextMiddleware",
    "request_id_var",
    "route_path",
]

logger = logging.getLogger("agentic_rag.api")

access_logger = logging.getLogger("agentic_rag.access")

REQUEST_ID_SCOPE_KEY = "agentic_rag.request_id"
BODY_EXCEEDED_SCOPE_KEY = "agentic_rag.body_exceeded"
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_VALID_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _problem_bytes(error: RagError) -> bytes:
    return json.dumps(error.to_problem()).encode()


def route_path(scope: Scope) -> str:
    """The path below the mount point. uvicorn's ``--root-path`` puts the prefix into ``scope["path"]``; the ASGI
    specification says it should not. Accept both so a prefix can never switch a protection off."""
    path: str = scope["path"]
    root: str = scope.get("root_path", "")
    if root and (path == root or path.startswith(root + "/")):
        path = path[len(root) :]
    return path or "/"


def _is_protected(path: str) -> bool:
    return path == "/v1" or path.startswith("/v1/") or path == "/metrics"


async def _send_problem(send: Send, error: RagError, *, extra_headers: list[tuple[bytes, bytes]] | None = None) -> None:
    body = _problem_bytes(error)
    headers = [(b"content-type", b"application/problem+json"), (b"content-length", str(len(body)).encode())]
    if error.retry_after is not None:
        headers.append((b"retry-after", str(max(1, round(error.retry_after))).encode()))
    headers.extend(extra_headers or [])
    await send({"type": "http.response.start", "status": error.http_status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class RequestContextMiddleware:
    """Request id, access log, metrics, and the last-resort 500 boundary (outermost user middleware)."""

    def __init__(self, app: ASGIApp, *, metrics: Metrics | None = None) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        inbound = MutableHeaders(scope=scope).get("x-request-id")
        request_id = inbound if inbound and _VALID_ID.fullmatch(inbound) else uuid.uuid4().hex
        scope[REQUEST_ID_SCOPE_KEY] = request_id
        token = request_id_var.set(request_id)
        started = False
        status = 500
        began = time.perf_counter()
        if self.metrics:
            self.metrics.request_started()

        async def send_with_id(message: Message) -> None:
            nonlocal started, status
            if message["type"] == "http.response.start":
                started = True
                status = int(message["status"])
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            logger.exception("unhandled error", extra={"request_id": request_id})
            if self.metrics:
                self.metrics.error(RagError.code)
            if not started:
                await _send_problem(send_with_id, RagError(request_id=request_id))
        finally:
            elapsed = time.perf_counter() - began
            matched = scope.get("route")
            # a template, never the raw path ("path_format" has no converters: /v1/documents/{document_id})
            route = getattr(matched, "path_format", None) or getattr(matched, "path", None) or "unmatched"
            if self.metrics:
                self.metrics.request_finished(scope["method"], route, status, elapsed)
            access_logger.info(
                "request",
                extra={
                    "event": "request",
                    "method": scope["method"],
                    "route": route,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 2),
                    "request_id": request_id,
                },
            )
            request_id_var.reset(token)


class AuthMiddleware:
    """Checks the API key before anything reads the request body or counts against the in-flight bound.

    Without this, FastAPI parses (and spools) a body before the route-level dependency runs, so an anonymous caller
    could make the server read 25 MB and could occupy every load-shedding slot. The route dependency stays as a second
    check. CORS preflights carry no credentials by design, so ``OPTIONS`` is left to the CORS layer.
    """

    def __init__(self, app: ASGIApp, *, config: ApiConfig, metrics: Metrics | None = None) -> None:
        self.app = app
        self.config = config
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] == "OPTIONS" or not _is_protected(route_path(scope)):
            await self.app(scope, receive, send)
            return
        authorization = MutableHeaders(scope=scope).get("authorization")
        if not is_authorized(self.config, bearer_token(authorization)):
            if self.metrics:
                self.metrics.error(AuthenticationFailed.code)
            error = AuthenticationFailed(request_id=scope.get(REQUEST_ID_SCOPE_KEY))
            await _send_problem(send, error, extra_headers=[(b"www-authenticate", b"Bearer")])
            return
        await self.app(scope, receive, send)


class LoadSheddingMiddleware:
    """Bounded in-flight work on ``/v1``: beyond the bound, answer 503 + Retry-After immediately (ADR-0009)."""

    def __init__(self, app: ASGIApp, *, max_inflight: int, metrics: Metrics | None = None) -> None:
        self.app = app
        self.max_inflight = max_inflight
        self.metrics = metrics
        self._inflight = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = route_path(scope) if scope["type"] == "http" else ""
        if scope["type"] != "http" or not (path == "/v1" or path.startswith("/v1/")):
            await self.app(scope, receive, send)
            return
        if self._inflight >= self.max_inflight:
            if self.metrics:
                self.metrics.shed()
                self.metrics.error(Overloaded.code)
            jitter = random.uniform(1, 3)  # noqa: S311 - retry jitter, not security
            error = Overloaded(retry_after=jitter, request_id=scope.get(REQUEST_ID_SCOPE_KEY))
            await _send_problem(send, error)
            return
        self._inflight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self._inflight -= 1


class BodyLimitMiddleware:
    """Rejects request bodies larger than ``max_bytes`` before they are spooled to disk."""

    def __init__(self, app: ASGIApp, *, max_bytes: int, metrics: Metrics | None = None) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Slack for multipart framing; the service enforces the exact file-size limit.
        limit = self.max_bytes + 64 * 1024
        declared = MutableHeaders(scope=scope).get("content-length")
        too_large = PayloadTooLarge(details={"max_bytes": self.max_bytes}, request_id=scope.get(REQUEST_ID_SCOPE_KEY))
        if declared is not None and declared.isdigit() and int(declared) > limit:
            if self.metrics:
                self.metrics.error(PayloadTooLarge.code)
            await _send_problem(send, too_large)
            return
        seen = 0
        exceeded = False
        started = False

        async def counting_receive() -> Message:
            nonlocal seen, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    exceeded = True
                    scope[BODY_EXCEEDED_SCOPE_KEY] = True  # the framework's own 422 for the cut-off body is not counted
                    if self.metrics:
                        self.metrics.error(PayloadTooLarge.code)
                    return {"type": "http.disconnect"}
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if exceeded:
                # Whatever the framework made of the truncated body, the caller gets a 413.
                if not started:
                    started = True
                    await _send_problem(send, too_large)
                return
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        finally:
            if exceeded and not started:
                await _send_problem(send, too_large)
