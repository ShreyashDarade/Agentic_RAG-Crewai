"""A CrewAI ``BaseLLM`` that delegates to our ``ChatModel`` port, so a crew uses whichever provider is configured."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import PrivateAttr

from agentic_rag.errors import DeadlineExceeded, ModelFailed, RagError
from agentic_rag.ports import ChatMessage, ChatModel

__all__ = ["PortLLM"]

_ROLES = {"system", "assistant", "user"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal parts: keep the text, drop the rest
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content)


class PortLLM(BaseLLM):
    """Runs on a CrewAI worker thread; the provider call is scheduled onto the application's event loop.

    The first failure is remembered and re-raised for every later call, so CrewAI's own retry loop cannot multiply
    provider calls (framework section 8: retries happen at one layer).
    """

    chat: Any
    loop: Any
    max_tokens: int = 700
    call_timeout: float = 60.0  # the crew's thread never waits on the application loop for longer than this
    _error: RagError | None = PrivateAttr(default=None)

    def cancel(self, error: RagError) -> None:
        """Stop the crew's thread from making further provider calls (it cannot be killed, only starved)."""
        self._error = error

    @property
    def failure(self) -> RagError | None:
        return self._error

    def supports_function_calling(self) -> bool:
        return False  # ReAct text format: portable across every ChatModel

    def get_context_window_size(self) -> int:
        return 16_000

    def call(
        self,
        messages: str | list[Any],
        tools: list[Any] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        if self._error is not None:
            raise self._error
        raw = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        converted = [
            ChatMessage(m["role"] if m.get("role") in _ROLES else "user", _text(m.get("content", ""))) for m in raw
        ]
        chat: ChatModel = self.chat
        future = asyncio.run_coroutine_threadsafe(chat.complete(converted, max_tokens=self.max_tokens), self.loop)
        try:
            completion = future.result(timeout=self.call_timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            self._error = DeadlineExceeded("a provider call exceeded its time limit")
            raise self._error from None
        except RagError as exc:
            self._error = exc
            raise
        except Exception as exc:
            self._error = ModelFailed()
            self._error.__cause__ = exc
            raise self._error from exc
        return self._apply_stop_words(completion.text)
