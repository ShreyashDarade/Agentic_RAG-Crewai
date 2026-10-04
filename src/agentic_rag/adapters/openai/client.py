from __future__ import annotations

import openai

__all__ = ["create_client"]


def create_client(*, api_key: str, base_url: str | None, timeout_seconds: float) -> openai.AsyncOpenAI:
    """The one place the SDK client is configured; server-internal calls retry at most once."""
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=1)
