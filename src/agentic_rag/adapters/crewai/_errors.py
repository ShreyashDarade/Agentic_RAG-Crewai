from __future__ import annotations

import math

from agentic_rag.errors import ConfigurationError, RagError, UpstreamRateLimited

__all__ = ["map_provider_error"]

_MAX_RETRY_AFTER = 60.0  # a provider asking for longer is not something a request can wait for


def _retry_after(value: object) -> float | None:
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return min(seconds, _MAX_RETRY_AFTER) if math.isfinite(seconds) and seconds >= 0 else None


def map_provider_error(exc: Exception, failure: type[RagError]) -> RagError:
    """CrewAI/LiteLLM/provider SDKs raise many types: classify by status code or class name; never copy their text."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    names = " ".join(c.__name__ for c in type(exc).__mro__)
    if status == 429 or "RateLimit" in names:
        headers = getattr(getattr(exc, "response", None), "headers", None) or {}
        return UpstreamRateLimited(retry_after=_retry_after(headers.get("retry-after")))
    if status in {401, 403} or "Authentication" in names or "PermissionDenied" in names:
        return ConfigurationError("the upstream provider rejected the configured credentials")
    return failure()
