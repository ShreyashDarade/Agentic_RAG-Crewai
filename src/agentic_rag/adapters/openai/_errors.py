from __future__ import annotations

import openai

from agentic_rag.errors import ConfigurationError, RagError, UpstreamRateLimited

__all__ = ["map_error"]


def _retry_after(exc: openai.APIStatusError) -> float | None:
    raw = exc.response.headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except ValueError:
        return None


def map_error(exc: openai.OpenAIError, failure: type[RagError]) -> RagError:
    """Translate an OpenAI SDK error; the SDK's text is never copied into the result."""
    if isinstance(exc, openai.RateLimitError):
        return UpstreamRateLimited(retry_after=_retry_after(exc))
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return ConfigurationError("the upstream provider rejected the configured credentials")
    return failure()
