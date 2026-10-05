from __future__ import annotations

from agentic_rag.errors import ConfigurationError, RagError, UpstreamRateLimited

__all__ = ["map_provider_error"]


def map_provider_error(exc: Exception, failure: type[RagError]) -> RagError:
    """CrewAI/LiteLLM/provider SDKs raise many types: classify by status code or class name; never copy their text."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    names = " ".join(c.__name__ for c in type(exc).__mro__)
    if status == 429 or "RateLimit" in names:
        retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
        try:
            return UpstreamRateLimited(retry_after=float(retry_after) if retry_after is not None else None)
        except (TypeError, ValueError):
            return UpstreamRateLimited()
    if status in {401, 403} or "Authentication" in names or "PermissionDenied" in names:
        return ConfigurationError("the upstream provider rejected the configured credentials")
    return failure()
