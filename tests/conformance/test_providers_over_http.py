"""Provider adapters over a real socket against a local OpenAI-compatible stub (not the real service)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from agentic_rag.errors import ConfigurationError, EmbeddingFailed, ModelFailed, UpstreamRateLimited
from agentic_rag.testing import ChatModelContract, EmbedderContract
from tests.stub_openai import DIM, StubOpenAI


@pytest.fixture(scope="module")
def stub() -> Iterator[StubOpenAI]:
    server = StubOpenAI().start()
    yield server
    server.stop()


@pytest.fixture(autouse=True)
def _reset(stub: StubOpenAI) -> None:
    stub.mode = "ok"
    stub.requests.clear()


@pytest.fixture(autouse=True)
def _wire(request: pytest.FixtureRequest, stub: StubOpenAI) -> None:
    if request.instance is not None:
        request.instance.stub = stub


def _openai_embedder(stub: StubOpenAI) -> Any:
    import openai

    from agentic_rag.adapters.openai import OpenAIEmbedder, create_client

    client = create_client(api_key="sk-test", base_url=stub.url, timeout_seconds=5)
    assert isinstance(client, openai.AsyncOpenAI)
    return OpenAIEmbedder(client, dimension=DIM)


class TestOpenAIEmbedderOverHttp(EmbedderContract):
    stub: StubOpenAI

    async def create(self) -> Any:
        return _openai_embedder(self.stub)

    async def create_failing(self) -> Any:
        self.stub.mode = "500"
        return _openai_embedder(self.stub)


class TestOpenAIChatOverHttp(ChatModelContract):
    stub: StubOpenAI

    async def create(self) -> Any:
        from agentic_rag.adapters.openai import OpenAIChatModel, create_client

        return OpenAIChatModel(create_client(api_key="sk-test", base_url=self.stub.url, timeout_seconds=5))

    async def create_failing(self) -> Any:
        self.stub.mode = "500"
        return await self.create()


class TestCrewAIChatModel(ChatModelContract):
    stub: StubOpenAI

    async def create(self) -> Any:
        pytest.importorskip("crewai")
        from agentic_rag.adapters.crewai import CrewAIChatModel

        return CrewAIChatModel("openai/gpt-4o-mini", api_key="sk-test", base_url=self.stub.url)

    async def create_failing(self) -> Any:
        self.stub.mode = "500"
        return await self.create()

    async def test_rate_limit_and_credentials_map_to_typed_errors_without_provider_text(self) -> None:
        from agentic_rag.testing.contracts import ChatMessage

        for mode, error in (("429", UpstreamRateLimited), ("401", ConfigurationError)):
            self.stub.mode = mode
            with pytest.raises(error) as err:
                await (await self.create()).complete([ChatMessage("user", "x")], max_tokens=5)
            assert "SECRET-DETAIL" not in json.dumps(err.value.to_problem())

    async def test_a_failing_provider_is_called_at_most_twice(self) -> None:
        from agentic_rag.testing.contracts import ChatMessage

        self.stub.mode = "500"
        with pytest.raises(ModelFailed):
            await (await self.create()).complete([ChatMessage("user", "x")], max_tokens=5)
        assert len(self.stub.requests) <= 2  # the first try plus at most one retry (framework section 8)

    async def test_options_reach_the_provider(self) -> None:
        from agentic_rag.testing.contracts import ChatMessage

        await (await self.create()).complete([ChatMessage("user", "x")], max_tokens=33, json_mode=True)
        body = self.stub.requests[-1]["body"]
        assert body["max_tokens"] == 33 or body.get("max_completion_tokens") == 33
        assert body["response_format"] == {"type": "json_object"}
        assert self.stub.requests[-1]["auth"] == "Bearer sk-test"


class TestCrewAIEmbedder(EmbedderContract):
    stub: StubOpenAI

    def _make(self, dimension: int = DIM) -> Any:
        pytest.importorskip("crewai")
        from agentic_rag.adapters.crewai import CrewAIEmbedder

        return CrewAIEmbedder(
            "openai",
            {"api_key": "sk-test", "api_base": self.stub.url, "model_name": "text-embedding-3-small"},
            dimension=dimension,
        )

    async def create(self) -> Any:
        return self._make()

    async def create_failing(self) -> Any:
        self.stub.mode = "500"
        return self._make()

    async def test_a_provider_returning_another_vector_size_is_an_error(self) -> None:
        with pytest.raises(EmbeddingFailed):
            await self._make(dimension=DIM + 1).embed_documents(["x"])

    async def test_model_identity_includes_provider_model_and_size(self) -> None:
        assert self._make().model_id == f"crewai:openai:text-embedding-3-small:{DIM}"

    async def test_an_unknown_provider_is_a_configuration_error(self) -> None:
        pytest.importorskip("crewai")
        from agentic_rag.adapters.crewai import CrewAIEmbedder

        with pytest.raises(ConfigurationError):
            CrewAIEmbedder("no-such-provider", {}, dimension=4)
