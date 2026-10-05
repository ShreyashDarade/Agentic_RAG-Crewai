"""OpenAI adapters against a stand-in for the HTTP API (failure injection only; no real key was used).

The embedding/chat endpoints here are a deterministic local fake of the documented JSON shapes.
Behaviour against the real API is therefore **unverified**.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import openai

from agentic_rag.adapters.openai import OpenAIChatModel, OpenAIEmbedder
from agentic_rag.errors import ConfigurationError, EmbeddingFailed, UpstreamRateLimited
from agentic_rag.testing import ChatModelContract, EmbedderContract

DIM = 12


def _handler(status: int = 200, headers: dict[str, str] | None = None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "SECRET-DETAIL", "type": "x"}}, headers=headers)
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            data = [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": [1.0 + len(t) % 7] + [float(len(t) % 5)] * (DIM - 1),
                }
                for i, t in enumerate(body["input"])
            ]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": data,
                    "model": body["model"],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    return httpx.MockTransport(handle)


def _client(status: int = 200, headers: dict[str, str] | None = None) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=_handler(status, headers)),
    )


class TestOpenAIEmbedder(EmbedderContract):
    async def create(self) -> Any:
        return OpenAIEmbedder(_client(), dimension=DIM)

    async def create_failing(self) -> Any:
        return OpenAIEmbedder(_client(500), dimension=DIM)

    async def test_rate_limit_keeps_retry_after_and_hides_provider_text(self) -> None:
        emb = OpenAIEmbedder(_client(429, {"retry-after": "7"}), dimension=DIM)
        try:
            await emb.embed_documents(["x"])
        except UpstreamRateLimited as exc:
            assert exc.retry_after == 7.0
            assert "SECRET-DETAIL" not in json.dumps(exc.to_problem())
        else:
            raise AssertionError("expected UpstreamRateLimited")

    async def test_bad_credentials_are_a_configuration_error(self) -> None:
        emb = OpenAIEmbedder(_client(401), dimension=DIM)
        try:
            await emb.embed_documents(["x"])
        except ConfigurationError as exc:
            assert "SECRET-DETAIL" not in json.dumps(exc.to_problem())
        else:
            raise AssertionError("expected ConfigurationError")

    async def test_provider_server_error_is_embedding_failed(self) -> None:
        try:
            await OpenAIEmbedder(_client(503), dimension=DIM).embed_query("x")
        except EmbeddingFailed:
            pass
        else:
            raise AssertionError("expected EmbeddingFailed")


class TestOpenAIChatModel(ChatModelContract):
    async def create(self) -> Any:
        return OpenAIChatModel(_client())

    async def create_failing(self) -> Any:
        return OpenAIChatModel(_client(500))
