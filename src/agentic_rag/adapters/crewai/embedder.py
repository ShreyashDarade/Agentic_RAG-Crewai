"""``Embedder`` over CrewAI's embedding providers (OpenAI, Azure, Bedrock, Cohere, Google, Ollama, Voyage, Jina,
Hugging Face, sentence-transformers, ONNX, Watsonx and a custom callable). The vector size must be declared and is
checked on every call: a provider that returns another size is an error, never a silently mixed index."""

from __future__ import annotations

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from crewai.rag.embeddings.factory import build_embedder

from agentic_rag.adapters.crewai._errors import map_provider_error
from agentic_rag.errors import ConfigurationError, EmbeddingFailed

__all__ = ["CrewAIEmbedder"]


class CrewAIEmbedder:
    def __init__(
        self,
        provider: str,
        config: dict[str, Any],
        *,
        dimension: int,
        factory: Callable[[dict[str, Any]], Any] = build_embedder,
    ) -> None:
        try:
            self._embed = factory({"provider": provider, "config": config})
        except Exception as exc:
            raise ConfigurationError(f"the CrewAI embedder {provider!r} could not be configured") from exc
        self._provider = provider
        self._model = str(config.get("model_name") or config.get("model") or "default")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return f"crewai:{self._provider}:{self._model}:{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            raw = await asyncio.to_thread(self._embed, list(texts))
        except Exception as exc:
            raise map_provider_error(exc, EmbeddingFailed) from exc
        vectors = [[float(x) for x in v] for v in raw]
        if len(vectors) != len(texts) or any(len(v) != self._dimension for v in vectors):
            raise EmbeddingFailed("the provider returned the wrong number or size of vectors")
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
