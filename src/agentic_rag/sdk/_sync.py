"""A blocking bridge over ONE background event loop (ADR-0004).

Calling a blocking method from a thread that is running an event loop raises ``UsageError`` instead of
stalling it. ``close()`` cancels in-flight calls and is idempotent.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from agentic_rag.errors import UsageError

__all__ = ["LoopThread"]

T = TypeVar("T")


class LoopThread:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._pending: set[concurrent.futures.Future[Any]] = set()
        self._closed = False
        self._thread = threading.Thread(target=self._main, name="agentic-rag-client", daemon=True)
        self._thread.start()
        self._ready.wait()

    def _main(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            raise UsageError(
                "a blocking client call was made inside a running event loop; use AsyncClient and await it"
            )
        with self._lock:
            if self._closed:
                coro.close()
                raise UsageError("the client is closed")
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            self._pending.add(future)
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            raise UsageError("the client was closed while this call was in flight") from None
        except BaseException:
            future.cancel()
            raise
        finally:
            with self._lock:
                self._pending.discard(future)

    def close(self, finalizer: Coroutine[Any, Any, None] | None = None) -> None:
        """Cancel in-flight calls, run ``finalizer`` on the loop, stop the loop. Safe to call twice."""
        with self._lock:
            if self._closed:
                if finalizer is not None:
                    finalizer.close()
                return
            self._closed = True
            pending = list(self._pending)
        for future in pending:
            future.cancel()
        try:
            if finalizer is not None:
                asyncio.run_coroutine_threadsafe(finalizer, self._loop).result(timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)
