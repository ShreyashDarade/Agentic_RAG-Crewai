"""The composition root (ADR-0001): the only module that imports adapters and chooses concrete classes.

``build_container`` reads the component *names* from settings, asks the registries for them, wires the
graph once and returns the :class:`Service`. Swapping an adapter is a settings change.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from agentic_rag.application import Service
from agentic_rag.application.retrieval import HybridRetriever
from agentic_rag.config import Settings
from agentic_rag.errors import ConfigurationError, RagError
from agentic_rag.ports import (
    AnswerPipeline,
    ChatModel,
    Chunker,
    ChunkScanner,
    DocumentCatalog,
    DocumentParser,
    Embedder,
    Healthcheck,
    LexicalIndex,
    Reranker,
    VectorSearcher,
    VectorWriter,
)
from agentic_rag.registry import Registries, load_plugins

__all__ = ["BuildContext", "Container", "build_container", "default_registries"]

logger = logging.getLogger(__name__)

# CrewAI sends anonymous usage telemetry unless told not to; the service never wants that.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

Closer = Callable[[], Awaitable[None]]


@dataclass
class BuildContext:
    """What factories may use: shared clients, closers, and the components built so far."""

    settings: Settings
    built: dict[str, Any] = field(default_factory=dict)
    closers: list[Closer] = field(default_factory=list)
    _shared: dict[str, Any] = field(default_factory=dict)

    def shared(self, key: str, make: Callable[[], Any], *, close: Closer | None = None) -> Any:
        if key not in self._shared:
            self._shared[key] = make()
            if close is not None:
                self.closers.append(close)
        return self._shared[key]


def _needs(extra: str, module: str) -> ConfigurationError:
    return ConfigurationError(f"{module!r} is not installed; install agentic-rag[{extra}]")


# -- built-in factories (imports are lazy so a missing extra fails only when its component is selected) --


def _milvus(s: Settings, ctx: BuildContext) -> Any:
    try:
        from agentic_rag.adapters.milvus import MilvusSettings, MilvusStore
    except ImportError as exc:
        raise _needs("engine", "pymilvus") from exc
    if not s.milvus_uri:
        raise ConfigurationError("AGENTIC_RAG_MILVUS_URI is required for the milvus vector store")
    store = MilvusStore(
        MilvusSettings(
            uri=s.milvus_uri,
            token=s.milvus_token.get_secret_value() if s.milvus_token else None,
            collection=s.milvus_collection,
            hnsw_m=s.hnsw_m,
            hnsw_ef_construction=s.hnsw_ef_construction,
            search_ef=s.search_ef,
            consistency_level=s.consistency_level,
        )
    )
    ctx.closers.append(store.aclose)
    return store


def _openai_client(s: Settings, ctx: BuildContext) -> Any:
    try:
        from agentic_rag.adapters.openai import create_client
    except ImportError as exc:
        raise _needs("engine", "openai") from exc
    if s.openai_api_key is None:
        raise ConfigurationError("AGENTIC_RAG_OPENAI_API_KEY is required for the openai components")
    api_key = s.openai_api_key.get_secret_value()
    holder: dict[str, Any] = {}

    def make() -> Any:
        holder["client"] = create_client(
            api_key=api_key, base_url=s.openai_base_url, timeout_seconds=s.openai_timeout_seconds
        )
        return holder["client"]

    async def close() -> None:
        await holder["client"].close()

    return ctx.shared("openai", make, close=close)


def _openai_embedder(s: Settings, ctx: BuildContext) -> Any:
    from agentic_rag.adapters.openai import OpenAIEmbedder

    return OpenAIEmbedder(_openai_client(s, ctx), model=s.embedding_model, dimension=s.embedding_dimension)


def _openai_chat(s: Settings, ctx: BuildContext) -> Any:
    from agentic_rag.adapters.openai import OpenAIChatModel

    return OpenAIChatModel(_openai_client(s, ctx), model=s.openai_chat_model)


def _bm25(s: Settings, ctx: BuildContext) -> Any:
    from agentic_rag.adapters.lexical_bm25 import Bm25Index

    return Bm25Index()


def _recursive_chunker(s: Settings, ctx: BuildContext) -> Any:
    from agentic_rag.adapters.chunker_recursive import RecursiveChunker

    return RecursiveChunker(max_chars=s.chunk_max_chars, overlap_chars=s.chunk_overlap_chars)


def _text_parser(s: Settings, ctx: BuildContext) -> Any:
    from agentic_rag.adapters.parser_text import TextParser

    return TextParser()


def _html_parser(s: Settings, ctx: BuildContext) -> Any:
    try:
        from agentic_rag.adapters.parser_html import HtmlParser
    except ImportError as exc:
        raise _needs("parsers", "beautifulsoup4/lxml") from exc
    return HtmlParser()


def _pdf_parser(s: Settings, ctx: BuildContext) -> Any:
    try:
        from agentic_rag.adapters.parser_pdf import PdfParser
    except ImportError as exc:
        raise _needs("parsers", "pymupdf") from exc
    return PdfParser(max_pages=s.pdf_max_pages)


def _docx_parser(s: Settings, ctx: BuildContext) -> Any:
    try:
        from agentic_rag.adapters.parser_docx import DocxParser
    except ImportError as exc:
        raise _needs("parsers", "python-docx") from exc
    return DocxParser()


def _direct_pipeline(s: Settings, ctx: BuildContext) -> Any:
    from agentic_rag.adapters.pipeline_direct import DirectPipeline

    return DirectPipeline(cast(ChatModel, ctx.built["chat_model"]), max_tokens=s.pipeline_max_tokens)


def default_registries() -> Registries:
    """Registries with every built-in component. Plug-ins add to (or replace) these."""
    regs = Registries()
    regs.vector_store.register("milvus", _milvus)
    regs.lexical_index.register("bm25", _bm25)
    regs.embedder.register("openai", _openai_embedder)
    regs.chat_model.register("openai", _openai_chat)
    regs.chunker.register("recursive", _recursive_chunker)
    regs.parser.register("text", _text_parser)
    regs.parser.register("html", _html_parser)
    regs.parser.register("pdf", _pdf_parser)
    regs.parser.register("docx", _docx_parser)
    regs.answer_pipeline.register("direct", _direct_pipeline)
    return regs


# -- the container -----------------------------------------------------------------------------------


class _LexicalHydration:
    """Rebuilds the in-memory lexical index from the vector store; reports readiness honestly."""

    def __init__(self, index: LexicalIndex, scanner: ChunkScanner, *, retry_seconds: float = 5.0) -> None:
        self._index = index
        self._scanner = scanner
        self._retry = retry_seconds
        self._done = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def check(self) -> None:
        if not self._done.is_set():
            raise ConfigurationError("the lexical index has not been rebuilt yet")

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="lexical-hydration")

    async def _run(self) -> None:
        while True:
            try:
                count = 0
                async for batch in self._scanner.scan(batch_size=500):
                    self._index.add(batch)
                    count += len(batch)
                logger.info("lexical index rebuilt from %d chunks", count)
                self._done.set()
                return
            except RagError as exc:
                logger.warning("lexical index rebuild failed (%s); retrying", exc.code)
                await asyncio.sleep(self._retry)

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


@dataclass
class Container:
    settings: Settings
    service: Service
    _closers: list[Closer] = field(default_factory=list)

    async def aclose(self) -> None:
        for closer in reversed(self._closers):
            try:
                await closer()
            except Exception:
                logger.exception("error while closing a resource")
        self._closers.clear()


async def build_container(settings: Settings, registries: Registries | None = None) -> Container:
    regs = registries or default_registries()
    load_plugins(regs, settings.plugins)
    ctx = BuildContext(settings)

    store = regs.vector_store.create(settings.vector_store, settings, ctx)
    embedder: Embedder = regs.embedder.create(settings.embedder, settings, ctx)
    ctx.built["chat_model"] = regs.chat_model.create(settings.chat_model, settings, ctx)
    chunker: Chunker = regs.chunker.create(settings.chunker, settings, ctx)
    parsers: list[DocumentParser] = [regs.parser.create(name, settings, ctx) for name in settings.parsers]
    pipeline: AnswerPipeline = regs.answer_pipeline.create(settings.answer_pipeline, settings, ctx)
    reranker: Reranker | None = regs.reranker.create(settings.reranker, settings, ctx) if settings.reranker else None
    lexical: LexicalIndex | None = (
        None if settings.lexical_index == "none" else regs.lexical_index.create(settings.lexical_index, settings, ctx)
    )

    retrieval = settings.to_retrieval()
    retriever = HybridRetriever(
        embedder=embedder,
        searcher=cast(VectorSearcher, store),
        lexical=lexical,
        reranker=reranker,
        config=retrieval,
    )
    health: dict[str, Healthcheck] = {"vector_store": cast(Healthcheck, store)}
    closers: list[Closer] = list(ctx.closers)
    if lexical is not None and settings.use_lexical:
        hydration = _LexicalHydration(lexical, cast(ChunkScanner, store))
        hydration.start()
        health["lexical_index"] = hydration
        closers.append(hydration.aclose)

    service = Service(
        embedder=embedder,
        writer=cast(VectorWriter, store),
        catalog=cast(DocumentCatalog, store),
        retriever=retriever,
        lexical=lexical if settings.use_lexical else None,
        parsers=parsers,
        chunker=chunker,
        pipeline=pipeline,
        limits=settings.to_limits(),
        retrieval=retrieval,
        health_checks=health,
    )
    return Container(settings=settings, service=service, _closers=closers)
