from __future__ import annotations

from collections.abc import Sequence

import openai

from agentic_rag.adapters.openai._errors import map_error
from agentic_rag.errors import EmbeddingFailed

__all__ = ["OpenAIEmbedder"]


class OpenAIEmbedder:
    def __init__(
        self,
        client: openai.AsyncOpenAI,
        *,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return f"openai:{self._model}:{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=list(texts), dimensions=self._dimension
            )
        except openai.OpenAIError as exc:
            raise map_error(exc, EmbeddingFailed) from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise EmbeddingFailed("the provider returned the wrong number of embeddings")
        return [list(item.embedding) for item in ordered]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
