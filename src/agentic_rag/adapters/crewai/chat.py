"""``ChatModel`` over ``crewai.LLM``: any provider CrewAI can route to (OpenAI, Anthropic, Azure, Gemini, Bedrock,
OpenAI-compatible servers such as Ollama or vLLM, and LiteLLM's long tail)."""

from __future__ import annotations

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from collections.abc import Sequence
from typing import Any

from crewai import LLM

from agentic_rag.adapters.crewai._errors import map_provider_error
from agentic_rag.blocking import PROVIDER_BACKLOG, PROVIDER_WORKERS, BlockingPool
from agentic_rag.errors import ModelFailed, Overloaded, RagError
from agentic_rag.ports import ChatMessage, Completion

__all__ = ["CrewAIChatModel"]


class CrewAIChatModel:
    """``model`` is CrewAI's provider-prefixed name, for example ``openai/gpt-4o-mini`` or ``ollama/llama3``."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        workers: int = PROVIDER_WORKERS,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout_seconds
        self._clients: dict[tuple[int, float, bool], LLM] = {}
        # Its own threads: a provider that hangs must not use up the threads parsing and store calls share, and a crew's
        # worker thread waits on this pool, so the two must never be the same pool.
        self._pool = BlockingPool("crewai-llm", workers=workers, backlog=PROVIDER_BACKLOG, saturated=Overloaded)

    def close(self) -> None:
        self._pool.close()

    def _client(self, max_tokens: int, temperature: float, json_mode: bool) -> LLM:
        key = (max_tokens, temperature, json_mode)
        if key not in self._clients:
            # max_retries=1: server-internal calls retry at most once (framework section 8); the default is two
            options: dict[str, Any] = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "max_retries": 1,
                "timeout": self._timeout,
            }
            if self._api_key:
                options["api_key"] = self._api_key
            if self._base_url:
                options["base_url"] = self._base_url
            if json_mode:
                options["response_format"] = {"type": "json_object"}
            self._clients[key] = LLM(model=self._model, **options)
        return self._clients[key]

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> Completion:
        payload: Any = [{"role": m.role, "content": m.content} for m in messages]
        client = self._client(max_tokens, temperature, json_mode)
        try:
            text = await self._pool.run(client.call, payload)
        except RagError:
            raise
        except Exception as exc:
            raise map_provider_error(exc, ModelFailed) from exc
        if not isinstance(text, str):
            raise ModelFailed("the provider returned a non-text response")
        return Completion(text=text)
