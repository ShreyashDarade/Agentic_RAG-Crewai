"""Review round 1 (composition root and configuration)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agentic_rag.adapters.lexical_bm25 import Bm25Index
from agentic_rag.api import ApiConfig, create_app
from agentic_rag.config import load_settings
from agentic_rag.container import _LexicalHydration, build_container, default_registries
from agentic_rag.errors import ConfigurationError
from agentic_rag.ports import Chunk
from agentic_rag.testing import FakeChatModel, FakeEmbedder, FakeVectorStore
from tests.conftest import API_KEY

ENV = {
    "AGENTIC_RAG_API_KEYS": API_KEY,
    "AGENTIC_RAG_VECTOR_STORE": "recording",
    "AGENTIC_RAG_EMBEDDER": "fake",
    "AGENTIC_RAG_CHAT_MODEL": "fake",
    "AGENTIC_RAG_EMBEDDING_DIMENSION": "32",
}


def _registries(closed: list[str], *, embedder_fails: bool = False):  # type: ignore[no-untyped-def]
    regs = default_registries()

    def store(s, ctx):  # type: ignore[no-untyped-def]
        async def close() -> None:
            closed.append("store")

        ctx.closers.append(close)
        return FakeVectorStore()

    def embedder(s, ctx):  # type: ignore[no-untyped-def]
        if embedder_fails:
            raise ConfigurationError("the embedder could not be built")
        return FakeEmbedder(s.embedding_dimension)

    regs.vector_store.register("recording", store)
    regs.embedder.register("fake", embedder)
    regs.chat_model.register("fake", lambda s, ctx: FakeChatModel(["{}"]))
    return regs


async def test_a_failed_build_closes_what_was_already_opened() -> None:
    closed: list[str] = []
    with pytest.raises(ConfigurationError):
        await build_container(load_settings(ENV), _registries(closed, embedder_fails=True))
    assert closed == ["store"]


async def test_a_reranker_setting_and_the_use_flag_must_agree() -> None:
    closed: list[str] = []
    with pytest.raises(ConfigurationError, match="RERANKER"):
        await build_container(load_settings({**ENV, "AGENTIC_RAG_USE_RERANKER": "true"}), _registries(closed))
    assert closed == ["store"]  # and the failure still releases the store
    regs = _registries([])
    regs.reranker.register("overlap", lambda s, ctx: __import__("agentic_rag.testing", fromlist=["x"]).FakeReranker())
    with pytest.raises(ConfigurationError, match="RERANKER"):
        await build_container(load_settings({**ENV, "AGENTIC_RAG_RERANKER": "overlap"}), regs)
    container = await build_container(
        load_settings({**ENV, "AGENTIC_RAG_RERANKER": "overlap", "AGENTIC_RAG_USE_RERANKER": "true"}), regs
    )
    await container.aclose()


# -- lexical hydration (review A6, B7) --------------------------------------------------------------------------


def _chunk(doc: str, text: str, index: int = 0) -> Chunk:
    return Chunk(id=f"ch_{doc}_{index}", document_id=doc, index=index, text=text, document_name=f"{doc}.txt")


class _ScriptedScanner:
    def __init__(self, batches: list[list[Chunk]], *, fail_first: BaseException | None = None) -> None:
        self.batches = batches
        self.fail_first = fail_first
        self.scans = 0
        self.gate: asyncio.Event | None = None
        self.paused = asyncio.Event()

    async def scan(self, *, batch_size: int) -> AsyncIterator[list[Chunk]]:
        self.scans += 1
        if self.fail_first is not None and self.scans == 1:
            raise self.fail_first
        for i, batch in enumerate(self.batches):
            if self.gate is not None and i == 1:
                self.paused.set()
                await self.gate.wait()
            yield batch


async def test_hydration_survives_any_exception_logs_it_and_finishes(caplog: pytest.LogCaptureFixture) -> None:
    index = Bm25Index()
    scanner = _ScriptedScanner([[_chunk("a", "apples are red")]], fail_first=RuntimeError("boom"))
    hydration = _LexicalHydration(index, scanner, retry_seconds=0.01)  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        hydration.start()
        for _ in range(200):
            try:
                await hydration.check()
                break
            except ConfigurationError:
                await asyncio.sleep(0.01)
        else:
            raise AssertionError("hydration never completed after a non-RagError failure")
    assert any(r.exc_info and "boom" in str(r.exc_info[1]) for r in caplog.records)  # logged with the cause
    assert [h.chunk.id for h in index.search("apples", top_k=3)] == ["ch_a_0"]
    await hydration.aclose()


async def test_a_delete_during_the_scan_is_not_undone_by_the_scan() -> None:
    index = Bm25Index()
    scanner = _ScriptedScanner([[_chunk("a", "apples are red")], [_chunk("b", "bananas are yellow")]])
    scanner.gate = asyncio.Event()
    hydration = _LexicalHydration(index, scanner)  # type: ignore[arg-type]
    hydration.start()
    await scanner.paused.wait()  # batch one is in; batch two has been read from the store but not yet added
    hydration.index.remove_document("b")  # the service deletes b meanwhile
    scanner.gate.set()
    for _ in range(200):
        try:
            await hydration.check()
            break
        except ConfigurationError:
            await asyncio.sleep(0.01)
    assert index.search("bananas", top_k=3) == []  # b stays deleted
    assert [h.chunk.id for h in index.search("apples", top_k=3)] == ["ch_a_0"]
    # once hydration is over the guard stops remembering: a re-ingest of b is indexed normally
    hydration.index.add([_chunk("b", "bananas are yellow")])
    assert [h.chunk.id for h in index.search("bananas", top_k=3)] == ["ch_b_0"]
    await hydration.aclose()


async def test_a_write_during_the_scan_wins_over_the_older_snapshot() -> None:
    index = Bm25Index()
    scanner = _ScriptedScanner([[_chunk("a", "apples")], [_chunk("c", "stale text about cherries")]])
    scanner.gate = asyncio.Event()
    hydration = _LexicalHydration(index, scanner)  # type: ignore[arg-type]
    hydration.start()
    await scanner.paused.wait()
    hydration.index.add([_chunk("c", "fresh text about cherries and plums")])  # re-ingested while the scan runs
    scanner.gate.set()
    for _ in range(200):
        try:
            await hydration.check()
            break
        except ConfigurationError:
            await asyncio.sleep(0.01)
    texts = [h.chunk.text for h in index.search("cherries", top_k=5)]
    assert texts == ["fresh text about cherries and plums"]
    await hydration.aclose()


# -- configuration (review B9) ----------------------------------------------------------------------------------


def test_the_sdk_api_key_variable_does_not_break_the_server_settings() -> None:
    settings = load_settings({**ENV, "AGENTIC_RAG_API_KEY": "client-side-key-0123456789"})
    assert [k.get_secret_value() for k in settings.api_keys] == [API_KEY]


def test_secrets_can_come_from_files(tmp_path: Path) -> None:
    keyfile = tmp_path / "keys"
    keyfile.write_text(API_KEY + "\n")
    env = {k: v for k, v in ENV.items() if k != "AGENTIC_RAG_API_KEYS"}
    settings = load_settings({**env, "AGENTIC_RAG_API_KEYS_FILE": str(keyfile)})
    assert [k.get_secret_value() for k in settings.api_keys] == [API_KEY]
    openai = tmp_path / "openai"
    openai.write_text("sk-test-0123456789\r\n")
    settings = load_settings({**ENV, "AGENTIC_RAG_OPENAI_API_KEY_FILE": str(openai)})
    assert settings.openai_api_key is not None and settings.openai_api_key.get_secret_value() == "sk-test-0123456789"


def test_a_secret_given_twice_or_from_a_missing_file_is_a_value_free_error(tmp_path: Path) -> None:
    keyfile = tmp_path / "keys"
    keyfile.write_text(API_KEY)
    with pytest.raises(ConfigurationError) as both:
        load_settings({**ENV, "AGENTIC_RAG_API_KEYS_FILE": str(keyfile)})
    assert "AGENTIC_RAG_API_KEYS" in str(both.value) and API_KEY not in str(both.value)
    env = {k: v for k, v in ENV.items() if k != "AGENTIC_RAG_API_KEYS"}
    with pytest.raises(ConfigurationError) as missing:
        load_settings({**env, "AGENTIC_RAG_API_KEYS_FILE": str(tmp_path / "nope")})
    assert "AGENTIC_RAG_API_KEYS_FILE" in str(missing.value)


def test_unauthenticated_mode_cannot_be_combined_with_keys() -> None:
    with pytest.raises(ConfigurationError, match="ALLOW_UNAUTHENTICATED"):
        load_settings({**ENV, "AGENTIC_RAG_ALLOW_UNAUTHENTICATED": "true"})
    with pytest.raises(ConfigurationError, match="ALLOW_UNAUTHENTICATED"):
        create_app(config=ApiConfig(api_keys=(API_KEY,), allow_unauthenticated=True))


def test_crewai_that_cannot_create_its_data_directory_is_a_typed_start_up_error() -> None:
    pytest.importorskip("crewai")
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import asyncio, os
        os.environ["XDG_DATA_HOME"] = "/proc/nope"  # mkdir below /proc fails even for root
        from agentic_rag.config import load_settings
        from agentic_rag.container import build_container, default_registries
        from agentic_rag.errors import ConfigurationError
        from agentic_rag.testing import FakeChatModel, FakeEmbedder, FakeVectorStore

        regs = default_registries()
        regs.vector_store.register("mem", lambda s, ctx: FakeVectorStore())
        regs.embedder.register("fake", lambda s, ctx: FakeEmbedder(s.embedding_dimension))
        regs.chat_model.register("fake", lambda s, ctx: FakeChatModel(["{}"]))
        env = {"AGENTIC_RAG_API_KEYS": "k" * 20, "AGENTIC_RAG_VECTOR_STORE": "mem", "AGENTIC_RAG_EMBEDDER": "fake",
               "AGENTIC_RAG_CHAT_MODEL": "fake", "AGENTIC_RAG_EMBEDDING_DIMENSION": "32",
               "AGENTIC_RAG_ANSWER_PIPELINE": "crewai"}
        try:
            asyncio.run(build_container(load_settings(env), regs))
        except ConfigurationError as exc:
            print("typed:", exc)
        except BaseException as exc:
            print("RAW:", type(exc).__name__)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
        check=False,
    )
    assert "typed:" in out.stdout and "XDG_DATA_HOME" in out.stdout, out.stdout + out.stderr
