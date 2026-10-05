"""Bounded worker pools for blocking calls (review round 1: a hung dependency must not take the process down).

``asyncio.to_thread`` shares one default executor between every caller and a cancelled await does not stop its
thread. So one dependency that accepts a connection and never answers used up the executor, and then parsing,
BM25 search and every other store call stalled behind it. A :class:`BlockingPool` gives each user of blocking
code its own small set of daemon threads and counts a call as running **until its thread really returns**,
not until the awaiting task is cancelled. When every worker and every queue slot is taken by calls that are not
finishing, the next call fails at once with the caller's own typed error instead of queueing for a deadline.

Standard library only.
"""

from __future__ import annotations

import asyncio
import contextvars
import queue
import threading
import weakref
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

__all__ = ["BlockingPool"]

T = TypeVar("T")

_Job = tuple[Future[Any], contextvars.Context, Callable[..., Any], tuple[Any, ...], dict[str, Any]]


def _worker(jobs: queue.SimpleQueue[_Job | None]) -> None:
    # Deliberately a module function holding only the queue, so an abandoned pool can be garbage collected.
    while True:
        job = jobs.get()
        if job is None:
            return
        _run(job)
        del job  # an idle worker must not keep the last call (and through it the pool) alive


def _run(job: _Job) -> None:
    future, context, fn, args, kwargs = job
    if not future.set_running_or_notify_cancel():
        return  # the caller gave up while it was still queued: it never ran
    try:
        result = context.run(fn, *args, **kwargs)
    except BaseException as exc:
        future.set_exception(exc)  # delivered to whoever awaits the call
        if not isinstance(exc, Exception):
            raise  # SystemExit and friends also end this worker, as they would any thread
    else:
        future.set_result(result)


class BlockingPool:
    """``workers`` daemon threads plus room for ``backlog`` waiting calls; beyond that ``saturated()`` is raised."""

    def __init__(self, name: str, *, workers: int, backlog: int, saturated: Callable[[], Exception]) -> None:
        if workers < 1 or backlog < 0:
            raise ValueError("a pool needs at least one worker and a non-negative backlog")
        self.name = name
        self._workers = workers
        self._capacity = workers + backlog
        self._saturated = saturated
        self._jobs: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._active = 0
        self._threads: list[threading.Thread] = []
        self._closed = False
        weakref.finalize(self, _stop, self._jobs, self._threads)

    @property
    def active(self) -> int:
        """Calls that have been accepted and whose thread has not yet finished (running or queued)."""
        return self._active

    async def run(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        future: Future[T] = Future()
        with self._lock:
            if self._closed:
                raise self._saturated()
            if self._active >= self._capacity:
                raise self._saturated()
            self._active += 1
            if len(self._threads) < self._workers and self._active > len(self._threads):
                thread = threading.Thread(
                    target=_worker, args=(self._jobs,), name=f"{self.name}-{len(self._threads)}", daemon=True
                )
                self._threads.append(thread)
                thread.start()
        future.add_done_callback(self._release)
        self._jobs.put((future, contextvars.copy_context(), fn, args, kwargs))
        return await asyncio.wrap_future(future)

    def close(self) -> None:
        """Stop idle workers. A worker stuck in a call is abandoned (it is a daemon thread)."""
        with self._lock:
            self._closed = True
        _stop(self._jobs, self._threads)

    def _release(self, _: Future[Any]) -> None:
        with self._lock:
            self._active -= 1


def _stop(jobs: queue.SimpleQueue[_Job | None], threads: list[threading.Thread]) -> None:
    for _ in threads:
        jobs.put(None)
