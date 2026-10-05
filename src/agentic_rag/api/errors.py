"""Exception handlers: every error leaves as RFC 9457 problem+json with a request id."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agentic_rag.api.middleware import BODY_EXCEEDED_SCOPE_KEY, REQUEST_ID_SCOPE_KEY
from agentic_rag.errors import MethodNotAllowed, NotFound, RagError, ValidationFailed

__all__ = ["install_error_handlers"]


def problem_response(error: RagError, request: Request) -> JSONResponse:
    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None and not request.scope.get(BODY_EXCEEDED_SCOPE_KEY):
        metrics.error(error.code)
    error.request_id = error.request_id or request.scope.get(REQUEST_ID_SCOPE_KEY)
    headers = {"WWW-Authenticate": "Bearer"} if error.http_status == 401 else {}
    if error.retry_after is not None:
        headers["Retry-After"] = str(max(1, round(error.retry_after)))
    return JSONResponse(
        error.to_problem(),
        status_code=error.http_status,
        media_type="application/problem+json",
        headers=headers,
    )


def install_error_handlers(app: FastAPI) -> None:
    async def on_rag_error(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, RagError)
        return problem_response(exc, request)

    async def on_validation(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, RequestValidationError)
        # Only the location and the message: never the submitted value (it may be a secret).
        errors = [{"loc": [str(p) for p in e["loc"]], "msg": str(e["msg"])} for e in exc.errors()]
        return problem_response(ValidationFailed(details={"errors": errors}), request)

    async def on_http(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, StarletteHTTPException)
        if exc.status_code == 404:
            return problem_response(NotFound(), request)
        if exc.status_code == 405:
            return problem_response(MethodNotAllowed(), request)
        return problem_response(ValidationFailed(), request)

    app.add_exception_handler(RagError, on_rag_error)
    app.add_exception_handler(RequestValidationError, on_validation)
    app.add_exception_handler(StarletteHTTPException, on_http)
