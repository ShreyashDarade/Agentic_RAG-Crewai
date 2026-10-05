"""``ChatModel`` over ``crewai.LLM``: any provider CrewAI can route to (OpenAI, Anthropic, Azure, Gemini, Bedrock,
OpenAI-compatible servers such as Ollama or vLLM, and LiteLLM's long tail)."""

from __future__ import annotations

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

import asyncio
from collections.abc import Sequence
from typing import Any

from crewai import LLM

from agentic_rag.adapters.crewai._errors import map_provider_error
from agentic_rag.errors import ModelFailed
from agentic_rag.ports import ChatMessage, Completion

__all__ = ["CrewAIChatModel"]


class CrewAIChatModel:
    """``model`` is CrewAI's provider-prefixed name, for example ``openai/gpt-4o-mini`` or ``ollama/llama3``."""

    def __init__(self, model: str, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._clients: dict[tuple[int, float, bool], LLM] = {}

    def _client(self, max_tokens: int, temperature: float, json_mode: bool) -> LLM:
        key = (max_tokens, temperature, json_mode)
        if key not in self._clients:
            # max_retries=1: server-internal calls retry at most once (framework section 8); the default is two
            options: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens, "max_retries": 1}
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
            text = await asyncio.to_thread(client.call, payload)
        except Exception as exc:
            raise map_provider_error(exc, ModelFailed) from exc
        if not isinstance(text, str):
            raise ModelFailed("the provider returned a non-text response")
        return Completion(text=text)
