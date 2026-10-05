"""Review round 1 (CrewAI adapter): no starvation, no stray files, bounded prompts, no silent fallback."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("crewai")

from agentic_rag.adapters.crewai import CrewAIChatModel, CrewPipeline
from agentic_rag.adapters.crewai._errors import map_provider_error
from agentic_rag.errors import DeadlineExceeded, ModelFailed, RagError, VectorStoreUnavailable
from agentic_rag.ports import ChunkFilter, Completion, ScoredChunk
from agentic_rag.testing import FakeChatModel, make_chunks
from tests.unit.test_crew_pipeline import Retrieval, react

ANSWER = react(json.dumps({"answer": "ok", "citations": []}))


class Hanging(FakeChatModel):
    """A provider that is slow enough to outlast every deadline in these tests."""

    async def complete(self, messages: Any, **kw: Any) -> Completion:
        self.calls.append(list(messages))
        await asyncio.sleep(30)
        raise AssertionError("unreachable")


# -- starvation (review C1) --------------------------------------------------------------------------------------


async def test_many_slow_crews_do_not_starve_the_default_executor() -> None:
    chat = Hanging()
    pipeline = CrewPipeline(chat, max_seconds=1, max_concurrent=16)  # more than the default executor has threads
    results = await asyncio.gather(
        *(pipeline.answer("q", retriever=Retrieval(), top_k=3) for _ in range(16)), return_exceptions=True
    )
    assert all(isinstance(r, DeadlineExceeded) for r in results), results
    started = time.monotonic()
    assert await asyncio.wait_for(asyncio.to_thread(lambda: "free"), 2) == "free"
    assert time.monotonic() - started < 1


async def test_more_crews_than_the_pool_allows_are_refused_not_queued_forever() -> None:
    from agentic_rag.errors import Overloaded

    pipeline = CrewPipeline(Hanging(), max_seconds=2, max_concurrent=1)
    results = await asyncio.gather(
        *(pipeline.answer("q", retriever=Retrieval(), top_k=3) for _ in range(6)), return_exceptions=True
    )
    assert any(isinstance(r, Overloaded) for r in results), results
    assert all(isinstance(r, DeadlineExceeded | Overloaded) for r in results), results


async def test_an_outer_deadline_also_stops_the_crew() -> None:
    class Counting(FakeChatModel):
        async def complete(self, messages: Any, **kw: Any) -> Completion:
            await asyncio.sleep(0.3)
            return await super().complete(messages, **kw)

    chat = Counting([ANSWER])
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.5):  # the service's request deadline, not the crew's own
            await CrewPipeline(chat, max_seconds=30).answer("q", retriever=Retrieval(), top_k=3)
    seen = len(chat.calls)
    await asyncio.sleep(1.5)
    assert len(chat.calls) <= seen + 1  # only the call already in flight may still land


# -- nothing is written to disk (review C2) ----------------------------------------------------------------------


async def test_a_request_writes_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("XDG_DATA_HOME", "CREWAI_STORAGE_DIR", "HOME"):
        monkeypatch.setenv(var, str(tmp_path))
    chat = FakeChatModel([react("[]"), react("ids"), react(json.dumps({"answer": "ok", "citations": ["ch_d_0"]}))])
    out = await CrewPipeline(chat).answer("q", retriever=Retrieval(), top_k=3)
    assert out.cited_chunk_ids == ("ch_d_0",)
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []  # CrewAI's own task-output SQLite file is not created


# -- bounded prompts (review C3) ---------------------------------------------------------------------------------


class Big:
    def __init__(self, n: int = 12, size: int = 3000) -> None:
        texts = [f"chunk {i} " + "x" * size for i in range(n)]
        self.hits = [ScoredChunk(c, 1.0 - i / 100) for i, c in enumerate(make_chunks("d", texts))]
        self.calls = 0

    async def retrieve(self, query: str, *, top_k: int, filter: ChunkFilter | None = None) -> list[ScoredChunk]:
        self.calls += 1
        return self.hits[:top_k]


async def test_the_search_tool_cannot_push_a_prompt_past_the_context_budget() -> None:
    budget = 6000
    chat = FakeChatModel(
        [
            react('["more"]'),
            'Thought: more\nAction: search_documents\nAction Input: {"query": "more"}',
            react("ids"),
            react(json.dumps({"answer": "ok", "citations": ["ch_d_0"]})),
        ]
    )
    await CrewPipeline(chat, max_context_chars=budget).answer("q", retriever=Big(), top_k=12)
    longest = max(sum(len(m.content) for m in call) for call in chat.calls)
    assert longest < budget * 3  # the evidence appears at most in a few prompts, each within the bound
    tool_turns = [m.content for call in chat.calls for m in call if "Observation" in m.content or "<chunk" in m.content]
    assert tool_turns
    assert all(len(text) < budget * 3 for text in tool_turns)


async def test_only_chunks_the_model_was_shown_can_be_cited() -> None:
    budget = 4000
    chat = FakeChatModel([react("[]"), react("ids"), react(json.dumps({"answer": "ok", "citations": ["ch_d_0"]}))])
    big = Big(n=12, size=3000)
    out = await CrewPipeline(chat, max_context_chars=budget).answer("q", retriever=big, top_k=12)
    shown = {h.chunk.id for h in out.retrieved}
    assert 0 < len(shown) < 12  # the rest were cut by the budget and never reached the model
    prompts = "\n".join(m.content for call in chat.calls for m in call)
    assert all(f'id="{cid}"' in prompts for cid in shown)


# -- no silent fallback (review C4) ------------------------------------------------------------------------------


async def test_a_store_failure_inside_the_tool_fails_the_request() -> None:
    class Flaky(Retrieval):
        async def retrieve(self, query: str, *, top_k: int, filter: ChunkFilter | None = None) -> list[ScoredChunk]:
            if self.calls:
                raise VectorStoreUnavailable()
            return await super().retrieve(query, top_k=top_k, filter=filter)

    chat = FakeChatModel(
        [
            react('["bananas colour"]'),
            'Thought: need more\nAction: search_documents\nAction Input: {"query": "bananas colour"}',
            react("ids"),
            react(json.dumps({"answer": "ok", "citations": ["ch_d_0"]})),
            react(json.dumps({"answer": "ok", "citations": ["ch_d_0"]})),
        ]
    )
    with pytest.raises(VectorStoreUnavailable):
        await CrewPipeline(chat).answer("q", retriever=Flaky(), top_k=3)


# -- provider errors (review C5) ---------------------------------------------------------------------------------


class _RateLimit(Exception):
    status_code = 429

    def __init__(self, retry_after: str) -> None:
        self.response = type("R", (), {"headers": {"retry-after": retry_after}})()


@pytest.mark.parametrize("value", ["nan", "inf", "-30", "86400", "x", "1e400", "5"])
def test_a_hostile_retry_after_never_breaks_error_serialisation(value: str) -> None:
    error = map_provider_error(_RateLimit(value), ModelFailed)
    assert isinstance(error, RagError)
    assert error.retry_after is None or 0 <= error.retry_after <= 60
    error.to_problem()
    assert max(1, round(error.retry_after or 0)) >= 1  # what the HTTP layer does with it


def test_the_error_model_itself_refuses_a_non_finite_retry_after() -> None:
    for bad in (float("nan"), float("inf"), -1.0):
        assert RagError(retry_after=bad).retry_after is None


# -- provider call has a timeout (review C1) ---------------------------------------------------------------------


def test_the_crewai_chat_model_passes_a_timeout_to_the_provider() -> None:
    model = CrewAIChatModel("openai/gpt-4o-mini", api_key="sk-test", timeout_seconds=12)
    client = model._client(100, 0.0, False)
    assert client.timeout == 12
