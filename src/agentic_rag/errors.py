"""The error model (ADR-0002).

Every failure a caller can observe is a :class:`RagError` subclass that defines its **own**
machine ``code``, ``http_status`` and ``public_message``. Raw dependency text is never part
of the public surface: wrap it with ``raise X(...) from exc`` and let logs carry the cause.

This module imports only the standard library, so the thin SDK can use it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "AuthenticationFailed",
    "ClientTimeout",
    "ConfigurationError",
    "ConnectionFailed",
    "DeadlineExceeded",
    "DependencyError",
    "DocumentEmpty",
    "DocumentNotFound",
    "DocumentParseFailed",
    "EmbeddingFailed",
    "FeatureDisabled",
    "IdempotencyKeyReused",
    "IndexIncompatible",
    "InvalidResponse",
    "LimitExceeded",
    "MethodNotAllowed",
    "ModelFailed",
    "ModelOutputInvalid",
    "NotFound",
    "Overloaded",
    "PayloadTooLarge",
    "RagError",
    "RagStatusError",
    "RateLimited",
    "RequestInFlight",
    "UnknownComponent",
    "UnsupportedFileType",
    "UpstreamRateLimited",
    "UsageError",
    "ValidationFailed",
    "VectorStoreError",
    "VectorStoreUnavailable",
    "catalog",
    "error_from_problem",
]

_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]+[A-Z0-9]$")
_MAX_CODE_LENGTH = 63
_CATALOG: dict[str, type[RagError]] = {}


class RagError(Exception):
    """Root of every error raised by this library.

    ``detail`` must be safe to show to a caller (it is serialised); never put dependency text
    in it. ``details`` carries structured, JSON-safe, non-secret data.
    """

    code: str = "INTERNAL_ERROR"
    http_status: int = 500
    public_message: str = "An internal error occurred."

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        own = cls.__dict__
        missing = [name for name in ("code", "http_status", "public_message") if name not in own]
        if missing:
            raise TypeError(
                f"{cls.__name__} must define its own {', '.join(missing)} "
                "(a subclass may not inherit its parent's wire identity)"
            )
        _register(cls)

    def __init__(
        self,
        detail: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(detail or self.public_message)
        self.detail = detail
        self.details: dict[str, Any] = dict(details or {})
        self.request_id = request_id
        self.retry_after = retry_after

    def to_problem(self) -> dict[str, Any]:
        """RFC 9457 body with the extension members ``code`` and ``request_id``."""
        body: dict[str, Any] = {
            "type": f"urn:agentic-rag:error:{self.code.lower()}",
            "title": self.public_message,
            "status": self.http_status,
            "code": self.code,
        }
        if self.detail:
            body["detail"] = self.detail
        if self.details:
            body["details"] = self.details
        if self.request_id:
            body["request_id"] = self.request_id
        if self.retry_after is not None:
            body["retry_after"] = self.retry_after
        return body


def _register(cls: type[RagError]) -> None:
    code = cls.__dict__["code"]
    if not isinstance(code, str) or not _CODE_PATTERN.match(code) or len(code) > _MAX_CODE_LENGTH:
        raise TypeError(f"{cls.__name__}.code {code!r} must match {_CODE_PATTERN.pattern}")
    if code in _CATALOG:
        raise TypeError(f"error code {code!r} already belongs to {_CATALOG[code].__name__}")
    _CATALOG[code] = cls


_CATALOG["INTERNAL_ERROR"] = RagError


def catalog() -> dict[str, type[RagError]]:
    """``code -> class`` for every error, built from the class hierarchy."""
    return dict(_CATALOG)


# -- caller mistakes (4xx) ----------------------------------------------------------------------


class ValidationFailed(RagError):
    code = "VALIDATION_FAILED"
    http_status = 422
    public_message = "The request is not valid."


class LimitExceeded(RagError):
    code = "LIMIT_EXCEEDED"
    http_status = 422
    public_message = "A configured limit was exceeded."


class AuthenticationFailed(RagError):
    code = "AUTHENTICATION_FAILED"
    http_status = 401
    public_message = "Authentication failed."


class FeatureDisabled(RagError):
    code = "FEATURE_DISABLED"
    http_status = 403
    public_message = "This operation is disabled on this server."


class NotFound(RagError):
    code = "NOT_FOUND"
    http_status = 404
    public_message = "The resource was not found."


class MethodNotAllowed(RagError):
    code = "METHOD_NOT_ALLOWED"
    http_status = 405
    public_message = "The method is not allowed for this resource."


class DocumentNotFound(RagError):
    code = "DOCUMENT_NOT_FOUND"
    http_status = 404
    public_message = "The document was not found."


class PayloadTooLarge(RagError):
    code = "PAYLOAD_TOO_LARGE"
    http_status = 413
    public_message = "The payload is larger than the configured limit."


class UnsupportedFileType(RagError):
    code = "UNSUPPORTED_FILE_TYPE"
    http_status = 415
    public_message = "This file type is not supported."


class DocumentParseFailed(RagError):
    code = "DOCUMENT_PARSE_FAILED"
    http_status = 422
    public_message = "The document could not be parsed."


class DocumentEmpty(RagError):
    code = "DOCUMENT_EMPTY"
    http_status = 422
    public_message = "The document contains no extractable text."


class IdempotencyKeyReused(RagError):
    code = "IDEMPOTENCY_KEY_REUSED"
    http_status = 422
    public_message = "The idempotency key was already used with a different request."


class RequestInFlight(RagError):
    code = "REQUEST_IN_FLIGHT"
    http_status = 409
    public_message = "A request with this idempotency key is still being processed."


class IndexIncompatible(RagError):
    code = "INDEX_INCOMPATIBLE"
    http_status = 409
    public_message = "The stored index was built with a different embedding model or size."


# -- capacity and time ---------------------------------------------------------------------------


class RateLimited(RagError):
    code = "RATE_LIMITED"
    http_status = 429
    public_message = "Too many requests."


class Overloaded(RagError):
    code = "OVERLOADED"
    http_status = 503
    public_message = "The server is overloaded; retry later."


class DeadlineExceeded(RagError):
    code = "DEADLINE_EXCEEDED"
    http_status = 504
    public_message = "The request exceeded its deadline."


# -- dependencies (5xx) --------------------------------------------------------------------------


class DependencyError(RagError):
    code = "DEPENDENCY_ERROR"
    http_status = 502
    public_message = "A dependency failed."


class VectorStoreUnavailable(DependencyError):
    code = "VECTOR_STORE_UNAVAILABLE"
    http_status = 503
    public_message = "The vector store is unavailable."


class VectorStoreError(DependencyError):
    code = "VECTOR_STORE_ERROR"
    http_status = 502
    public_message = "The vector store rejected the operation."


class EmbeddingFailed(DependencyError):
    code = "EMBEDDING_FAILED"
    http_status = 502
    public_message = "Computing embeddings failed."


class UpstreamRateLimited(DependencyError):
    code = "UPSTREAM_RATE_LIMITED"
    http_status = 429
    public_message = "An upstream provider is rate limiting this server."


class ModelFailed(DependencyError):
    code = "MODEL_FAILED"
    http_status = 502
    public_message = "The language model call failed."


class ModelOutputInvalid(DependencyError):
    code = "MODEL_OUTPUT_INVALID"
    http_status = 502
    public_message = "The language model returned output that failed validation."


# -- server configuration ------------------------------------------------------------------------


class ConfigurationError(RagError):
    code = "CONFIGURATION_ERROR"
    http_status = 500
    public_message = "The server is misconfigured."


class UnknownComponent(ConfigurationError):
    code = "UNKNOWN_COMPONENT"
    http_status = 500
    public_message = "No component is registered under that name."


# -- client side (never on the wire: http_status 0) ----------------------------------------------


class UsageError(RagError):
    """The SDK was used incorrectly (for example a blocking call inside a running event loop)."""

    code = "USAGE_ERROR"
    http_status = 0
    public_message = "The client was used incorrectly."


class ConnectionFailed(RagError):
    code = "CONNECTION_FAILED"
    http_status = 0
    public_message = "Could not reach the server."


class ClientTimeout(RagError):
    code = "CLIENT_TIMEOUT"
    http_status = 0
    public_message = "The client gave up waiting for the server."


class InvalidResponse(RagError):
    """The server answered, but the body did not match the contract this client understands."""

    code = "INVALID_RESPONSE"
    http_status = 0
    public_message = "The server returned a response this client could not understand."


class RagStatusError(RagError):
    """A well-formed error from a server whose code this client does not know.

    Keeps the server's ``code`` and ``http_status`` so newer servers stay usable by older clients.
    """

    code = "UNKNOWN_ERROR"
    http_status = 0
    public_message = "The server returned an error."

    def __init__(
        self,
        detail: str | None = None,
        *,
        server_code: str | None = None,
        status: int = 0,
        title: str | None = None,
        details: Mapping[str, Any] | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            detail or title,
            details=details,
            request_id=request_id,
            retry_after=retry_after,
        )
        if server_code is not None:
            self.code = server_code
        self.http_status = status
        if title:
            self.public_message = title


def _status_of(value: object, *, fallback: int) -> int:
    """An HTTP status from untrusted JSON: only an integer in the valid range counts; anything else is the fallback."""
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    return fallback


def _seconds_of(value: object) -> float | None:
    """A finite, non-negative number of seconds from untrusted JSON, or ``None``."""
    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def error_from_problem(body: Mapping[str, Any], *, status: int | None = None) -> RagError:
    """Rebuild the exception a server serialised with :meth:`RagError.to_problem`.

    A known code yields the same class the server raised; an unknown code yields
    :class:`RagStatusError` that keeps the code, status, request id and details.
    """
    raw_code = body.get("code")
    code = raw_code if isinstance(raw_code, str) else ""
    http_status = _status_of(body.get("status"), fallback=status or 0)
    detail = body.get("detail")
    details = body.get("details")
    request_id = body.get("request_id")
    common: dict[str, Any] = {
        "details": details if isinstance(details, Mapping) else None,
        "request_id": request_id if isinstance(request_id, str) else None,
        "retry_after": _seconds_of(body.get("retry_after")),
    }
    known = _CATALOG.get(code)
    if known is not None and known is not RagStatusError and known.http_status != 0:
        return known(detail if isinstance(detail, str) else None, **common)
    return RagStatusError(
        detail if isinstance(detail, str) else None,
        server_code=code or None,
        status=http_status,
        title=body.get("title") if isinstance(body.get("title"), str) else None,
        **common,
    )
