"""The composition root (ADR-0001): the only module that imports adapters and chooses concrete classes.

``build_container`` reads the component *names* from settings, asks the registries for them, wires the
graph once and returns the :class:`Service`. Swapping an adapter is a settings change.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from agentic_rag.application import Service
from agentic_rag.application.retrieval import HybridRetriever
from agentic_rag.config import Settings
from agentic_rag.errors import ConfigurationError
from agentic_rag.ports import (
    AnswerPipeline,
    ChatModel,
    Chunk,
    Chunker,
    ChunkFilter,
    ChunkScanner,
    DocumentCatalog,
    DocumentParser,
    Embedder,
    Healthcheck,
    LexicalIndex,
    Reranker,
    ScoredChunk,
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


def _sync_closer(close: Callable[[], None]) -> Closer:
    async def closer() -> None:
        close()

    return closer


def _crewai_data_dir() -> ConfigurationError:
    return ConfigurationError("crewai could not create its data directory; point XDG_DATA_HOME at a writable path")


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
            timeout_seconds=s.store_timeout_seconds,
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


def _chroma(s: Settings, ctx: BuildContext) -> Any:
    try:
        from agentic_rag.adapters.chroma import ChromaSettings, ChromaStore
    except ImportError as exc:
        raise _needs("chroma", "chromadb") from exc
    if (s.chroma_path is None) == (s.chroma_url is None):
        raise ConfigurationError("set exactly one of AGENTIC_RAG_CHROMA_PATH and AGENTIC_RAG_CHROMA_URL")
    store = ChromaStore(ChromaSettings(path=s.chroma_path, url=s.chroma_url, collection=s.chroma_collection))
    ctx.closers.append(store.aclose)
    return store


def _qdrant(s: Settings, ctx: BuildContext) -> Any:
    try:
        from agentic_rag.adapters.qdrant import QdrantSettings, QdrantStore
    except ImportError as exc:
        raise _needs("qdrant", "qdrant-client") from exc
    if not s.qdrant_location:
        raise ConfigurationError("AGENTIC_RAG_QDRANT_LOCATION is required for the qdrant vector store")
    key = s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None
    store = QdrantStore(
        QdrantSettings(
            location=s.qdrant_location,
            api_key=key,
            collection=s.qdrant_collection,
            timeout_seconds=s.store_timeout_seconds,
        )
    )
    ctx.closers.append(store.aclose)
    return store


def _crewai_chat(s: Settings, ctx: BuildContext) -> Any:
    try:
        from agentic_rag.adapters.crewai import CrewAIChatModel
    except ImportError as exc:
        raise _needs("crewai", "crewai") from exc
    except OSError as exc:  # importing crewai creates a data directory under XDG_DATA_HOME
        raise _crewai_data_dir() from exc
    if not s.crewai_llm_model:
        raise ConfigurationError("AGENTIC_RAG_CREWAI_LLM_MODEL is required for the crewai chat model")
    key = s.crewai_llm_api_key.get_secret_value() if s.crewai_llm_api_key else None
    chat = CrewAIChatModel(
        s.crewai_llm_model,
        api_key=key,
        base_url=s.crewai_llm_base_url,
        timeout_seconds=s.crewai_provider_timeout_seconds,
        workers=max(8, s.crewai_max_concurrent),
    )
    ctx.closers.append(_sync_closer(chat.close))
    return chat


def _crewai_embedder(s: Settings, ctx: BuildContext) -> Any:
    try:
        from agentic_rag.adapters.crewai import CrewAIEmbedder
    except ImportError as exc:
        raise _needs("crewai", "crewai") from exc
    except OSError as exc:  # importing crewai creates a data directory under XDG_DATA_HOME
        raise _crewai_data_dir() from exc
    if not s.crewai_embedder_provider:
        raise ConfigurationError("AGENTIC_RAG_CREWAI_EMBEDDER_PROVIDER is required for the crewai embedder")
    try:
        options = json.loads(s.crewai_embedder_options)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("AGENTIC_RAG_CREWAI_EMBEDDER_OPTIONS is not valid JSON") from exc
    if not isinstance(options, dict):
        raise ConfigurationError("AGENTIC_RAG_CREWAI_EMBEDDER_OPTIONS must be a JSON object")
    config: dict[str, Any] = dict(options)
    if s.crewai_embedder_model:
        config["model_name"] = s.crewai_embedder_model
    if s.crewai_embedder_base_url:
        config["api_base"] = s.crewai_embedder_base_url
    if s.crewai_embedder_api_key:
        config["api_key"] = s.crewai_embedder_api_key.get_secret_value()
    embedder = CrewAIEmbedder(
        s.crewai_embedder_provider, config, dimension=s.embedding_dimension, workers=max(8, s.crewai_max_concurrent)
    )
    ctx.closers.append(_sync_closer(embedder.close))
    return embedder


def _crewai_pipeline(s: Settings, ctx: BuildContext) -> Any:
    try:
        from agentic_rag.adapters.crewai import CrewPipeline
    except ImportError as exc:
        raise _needs("crewai", "crewai") from exc
    except OSError as exc:  # importing crewai creates a data directory under XDG_DATA_HOME
        raise _crewai_data_dir() from exc
    pipeline = CrewPipeline(
        cast(ChatModel, ctx.built["chat_model"]),
        max_tokens=s.pipeline_max_tokens,
        max_iter=s.crewai_max_iter,
        max_seconds=s.crewai_max_seconds,
        max_concurrent=s.crewai_max_concurrent,
    )
    ctx.closers.append(_sync_closer(pipeline.close))
    return pipeline


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
    regs.vector_store.register("chroma", _chroma)
    regs.vector_store.register("qdrant", _qdrant)
    regs.lexical_index.register("bm25", _bm25)
    regs.embedder.register("openai", _openai_embedder)
    regs.chat_model.register("openai", _openai_chat)
    regs.chat_model.register("crewai", _crewai_chat)
    regs.embedder.register("crewai", _crewai_embedder)
    regs.chunker.register("recursive", _recursive_chunker)
    regs.parser.register("text", _text_parser)
    regs.parser.register("html", _html_parser)
    regs.parser.register("pdf", _pdf_parser)
    regs.parser.register("docx", _docx_parser)
    regs.answer_pipeline.register("direct", _direct_pipeline)
    regs.answer_pipeline.register("crewai", _crewai_pipeline)
    return regs


# -- the container -----------------------------------------------------------------------------------


class _GuardedIndex:
    """The lexical index as the service and the retriever see it while it is being rebuilt from the store.

    The rebuild reads the store in batches while requests keep deleting and re-ingesting. A batch that was read before
    a delete and added after it would bring the deleted text back, and an old snapshot of a re-ingested document would
    overwrite the new one. So while the rebuild runs, every document the service touches is remembered, and the rebuild
    skips those: whatever the service did is newer than what the scan read. All operations share one lock, so a scan
    batch cannot slip in between the check and the write.
    """

    def __init__(self, inner: LexicalIndex) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._touched: set[str] | None = None

    def begin(self) -> None:
        with self._lock:
            self._touched = set() if self._touched is None else self._touched

    def end(self) -> None:
        with self._lock:
            self._touched = None

    def add(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            if self._touched is not None:
                self._touched.update(c.document_id for c in chunks)
            self._inner.add(chunks)

    def remove_document(self, document_id: str) -> None:
        with self._lock:
            if self._touched is not None:
                self._touched.add(document_id)
            self._inner.remove_document(document_id)

    def search(self, query: str, *, top_k: int, filter: ChunkFilter | None = None) -> list[ScoredChunk]:
        return self._inner.search(query, top_k=top_k, filter=filter)

    def add_scanned(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            touched = self._touched or set()
            fresh = [c for c in chunks if c.document_id not in touched]
            if fresh:
                self._inner.add(fresh)


class _LexicalHydration:
    """Rebuilds the in-memory lexical index from the vector store; reports readiness honestly."""

    def __init__(self, index: LexicalIndex, scanner: ChunkScanner, *, retry_seconds: float = 5.0) -> None:
        self.index = _GuardedIndex(index)  # what the service and the retriever must use
        self._scanner = scanner
        self._retry = retry_seconds
        self._done = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def check(self) -> None:
        if not self._done.is_set():
            raise ConfigurationError("the lexical index has not been rebuilt yet")

    def start(self) -> None:
        self.index.begin()
        self._task = asyncio.create_task(self._run(), name="lexical-hydration")

    async def _run(self) -> None:
        while True:
            try:
                count = 0
                async for batch in self._scanner.scan(batch_size=500):
                    self.index.add_scanned(batch)
                    count += len(batch)
                self.index.end()
                logger.info("lexical index rebuilt from %d chunks", count)
                self._done.set()
                return
            except asyncio.CancelledError:
                raise
            except Exception:  # any failure, not only RagError: a dead task would leave /readyz failing forever
                logger.exception("lexical index rebuild failed; retrying", extra={"event": "lexical_rebuild_failed"})
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
    try:
        return _assemble(settings, regs, ctx)
    except BaseException:
        # Factories register their closers as they open things; a later factory failing must not leak them.
        for closer in reversed(ctx.closers):
            try:
                await closer()
            except Exception:
                logger.exception("error while closing a resource after a failed start-up")
        raise


def _assemble(settings: Settings, regs: Registries, ctx: BuildContext) -> Container:
    store = regs.vector_store.create(settings.vector_store, settings, ctx)
    embedder: Embedder = regs.embedder.create(settings.embedder, settings, ctx)
    ctx.built["chat_model"] = regs.chat_model.create(settings.chat_model, settings, ctx)
    chunker: Chunker = regs.chunker.create(settings.chunker, settings, ctx)
    parsers: list[DocumentParser] = [regs.parser.create(name, settings, ctx) for name in settings.parsers]
    pipeline: AnswerPipeline = regs.answer_pipeline.create(settings.answer_pipeline, settings, ctx)
    reranker: Reranker | None = regs.reranker.create(settings.reranker, settings, ctx) if settings.reranker else None
    if settings.use_reranker != (reranker is not None):
        raise ConfigurationError("AGENTIC_RAG_USE_RERANKER and AGENTIC_RAG_RERANKER must be set together")
    lexical: LexicalIndex | None = (
        None if settings.lexical_index == "none" else regs.lexical_index.create(settings.lexical_index, settings, ctx)
    )

    hydration: _LexicalHydration | None = None
    if lexical is not None and settings.use_lexical:
        hydration = _LexicalHydration(lexical, cast(ChunkScanner, store))
        lexical = hydration.index
    elif not settings.use_lexical:
        lexical = None

    retrieval = settings.to_retrieval()
    retriever = HybridRetriever(
        embedder=embedder,
        searcher=cast(VectorSearcher, store),
        lexical=lexical,
        reranker=reranker,
        config=retrieval,
    )
    health: dict[str, Healthcheck] = {"vector_store": cast(Healthcheck, store)}
    if hydration is not None:
        health["lexical_index"] = hydration
    service = Service(
        embedder=embedder,
        writer=cast(VectorWriter, store),
        catalog=cast(DocumentCatalog, store),
        retriever=retriever,
        lexical=lexical,
        parsers=parsers,
        chunker=chunker,
        pipeline=pipeline,
        limits=settings.to_limits(),
        retrieval=retrieval,
        health_checks=health,
    )

    async def close_threads() -> None:
        service.close()
        retriever.close()

    closers: list[Closer] = [*ctx.closers, close_threads]
    if hydration is not None:
        hydration.start()  # last: nothing after this can fail, so the task is never orphaned
        closers.append(hydration.aclose)
    return Container(settings=settings, service=service, _closers=closers)
