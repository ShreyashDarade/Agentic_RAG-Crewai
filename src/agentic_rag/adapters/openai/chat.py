from __future__ import annotations

from collections.abc import Sequence

import openai
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONObject, ResponseFormatText

from agentic_rag.adapters.openai._errors import map_error
from agentic_rag.errors import ModelFailed
from agentic_rag.ports import ChatMessage, Completion

__all__ = ["OpenAIChatModel"]


class OpenAIChatModel:
    def __init__(self, client: openai.AsyncOpenAI, *, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self._model = model

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> Completion:
        payload: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": m.content}
            if m.role == "system"
            else {"role": "assistant", "content": m.content}
            if m.role == "assistant"
            else {"role": "user", "content": m.content}
            for m in messages
        ]
        response_format: ResponseFormatText | ResponseFormatJSONObject = (
            {"type": "json_object"} if json_mode else {"type": "text"}
        )
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=payload,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
                stream=False,
            )
        except openai.OpenAIError as exc:
            raise map_error(exc, ModelFailed) from exc
        if not response.choices:
            raise ModelFailed("the provider returned no choices")
        usage = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
