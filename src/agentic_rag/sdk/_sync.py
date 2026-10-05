"""A blocking bridge over ONE background event loop (ADR-0004).

Calling a blocking method from a thread that is running an event loop raises ``UsageError`` instead of
stalling it. ``close()`` cancels in-flight calls and is idempotent.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import weakref
from collections.abc import Coroutine
from typing import Any, TypeVar

from agentic_rag.errors import UsageError

__all__ = ["LoopThread"]

T = TypeVar("T")


def _main(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    # A module function, so the thread does not keep its LoopThread alive: a dropped client can then be collected.
    asyncio.set_event_loop(loop)
    loop.call_soon(ready.set)
    try:
        loop.run_forever()
    finally:
        loop.close()


def _stop(loop: asyncio.AbstractEventLoop) -> None:
    if not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)


class LoopThread:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._pending: set[concurrent.futures.Future[Any]] = set()
        self._closed = False
        self._pid = os.getpid()
        self._thread = threading.Thread(
            target=_main, args=(self._loop, self._ready), name="agentic-rag-client", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        # A client that is dropped without close() must not leave a thread and three file descriptors behind.
        self._finalizer = weakref.finalize(self, _stop, self._loop)

    @property
    def closed(self) -> bool:
        return self._closed

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        if os.getpid() != self._pid:
            coro.close()  # the loop thread does not exist in a forked child: waiting for it would hang forever
            raise UsageError("this client was created before fork(); create a new Client in the child process")
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
        if os.getpid() != self._pid:  # a forked child owns nothing here
            if finalizer is not None:
                finalizer.close()
            self._finalizer.detach()
            return
        for future in pending:
            future.cancel()
        try:
            if finalizer is not None:
                asyncio.run_coroutine_threadsafe(finalizer, self._loop).result(timeout=10)
        finally:
            self._finalizer()  # stops the loop (once)
            self._thread.join(timeout=10)
