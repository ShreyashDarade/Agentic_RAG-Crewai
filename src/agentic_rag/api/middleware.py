"""Pure-ASGI middleware (``BaseHTTPMiddleware`` would break contextvars and streaming)."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agentic_rag.errors import PayloadTooLarge, RagError

__all__ = ["REQUEST_ID_SCOPE_KEY", "BodyLimitMiddleware", "RequestContextMiddleware", "request_id_var"]

logger = logging.getLogger("agentic_rag.api")

REQUEST_ID_SCOPE_KEY = "agentic_rag.request_id"
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_VALID_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _problem_bytes(error: RagError) -> bytes:
    return json.dumps(error.to_problem()).encode()


async def _send_problem(send: Send, error: RagError) -> None:
    body = _problem_bytes(error)
    headers = [(b"content-type", b"application/problem+json"), (b"content-length", str(len(body)).encode())]
    if error.retry_after is not None:
        headers.append((b"retry-after", str(max(1, round(error.retry_after))).encode()))
    await send({"type": "http.response.start", "status": error.http_status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class RequestContextMiddleware:
    """Request id on every request/response/log line, and the last-resort 500 boundary."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        inbound = MutableHeaders(scope=scope).get("x-request-id")
        request_id = inbound if inbound and _VALID_ID.fullmatch(inbound) else uuid.uuid4().hex
        scope[REQUEST_ID_SCOPE_KEY] = request_id
        token = request_id_var.set(request_id)
        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            logger.exception("unhandled error", extra={"request_id": request_id})
            if not started:
                await _send_problem(send_with_id, RagError(request_id=request_id))
        finally:
            request_id_var.reset(token)


class BodyLimitMiddleware:
    """Rejects request bodies larger than ``max_bytes`` before they are spooled to disk."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Slack for multipart framing; the service enforces the exact file-size limit.
        limit = self.max_bytes + 64 * 1024
        declared = MutableHeaders(scope=scope).get("content-length")
        too_large = PayloadTooLarge(
            details={"max_bytes": self.max_bytes}, request_id=scope.get(REQUEST_ID_SCOPE_KEY)
        )
        if declared is not None and declared.isdigit() and int(declared) > limit:
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
