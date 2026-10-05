"""A store that accepts the connection and never answers must cost one typed error, not the process (review A1)."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import Iterator

import pytest

from agentic_rag.blocking import STORE_WORKERS
from agentic_rag.errors import VectorStoreUnavailable
from agentic_rag.ports import ChunkFilter


@pytest.fixture
def hung_server() -> Iterator[int]:
    """Accepts TCP connections and then says nothing."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    stop = threading.Event()
    held: list[socket.socket] = []

    def accept() -> None:
        listener.settimeout(0.1)
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            held.append(conn)

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield int(listener.getsockname()[1])
    stop.set()
    thread.join(2)
    for conn in held:
        conn.close()
    listener.close()


async def test_qdrant_gives_up_after_its_timeout_with_a_typed_error(hung_server: int) -> None:
    pytest.importorskip("qdrant_client")
    from agentic_rag.adapters.qdrant import QdrantSettings, QdrantStore

    store = QdrantStore(QdrantSettings(location=f"http://127.0.0.1:{hung_server}", timeout_seconds=1))
    started = time.monotonic()
    with pytest.raises(VectorStoreUnavailable):
        await store.check()
    assert time.monotonic() - started < 5
    await store.aclose()


async def test_milvus_gives_up_after_its_timeout_with_a_typed_error(hung_server: int) -> None:
    pytest.importorskip("pymilvus")
    from agentic_rag.adapters.milvus import MilvusSettings, MilvusStore

    store = MilvusStore(MilvusSettings(uri=f"http://127.0.0.1:{hung_server}", timeout_seconds=1))
    started = time.monotonic()
    with pytest.raises(VectorStoreUnavailable):
        await store.check()
    assert time.monotonic() - started < 10
    await store.aclose()


async def test_chroma_calls_that_never_return_strand_only_the_stores_own_threads(hung_server: int) -> None:
    pytest.importorskip("chromadb")
    from agentic_rag.adapters.chroma import ChromaSettings, ChromaStore

    store = ChromaStore(ChromaSettings(url=f"http://127.0.0.1:{hung_server}"))

    async def one() -> None:
        async with asyncio.timeout(0.3):
            await store.search([0.1] * 8, top_k=3, filter=ChunkFilter())

    results = await asyncio.gather(*(one() for _ in range(STORE_WORKERS)), return_exceptions=True)
    assert all(isinstance(r, TimeoutError) for r in results)  # each request hit its own deadline
    started = time.monotonic()
    assert await asyncio.to_thread(lambda: "free") == "free"  # the shared executor is untouched
    assert time.monotonic() - started < 1
